import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTSeg(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes_cls, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # the embedding size of each patch (including multi heads)
        self.dim = dim

        # the height and width of the whole image
        image_height, image_width = pair(image_size)
        # the height and width of each patch
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # the number of patches in one side
        self.num_patches_side = image_height // patch_height
        # the number of patches in the whole image
        self.num_patches = self.num_patches_side * self.num_patches_side

        # the original vector size of each patch before encoder
        self.img_patch_dim = channels * patch_height * patch_width
        self.double_img_patch_dim = 2 * channels * patch_height * patch_width
        self.binary_patch_dim = 1 * patch_height * patch_width

        # convert the original patch vector to embedding
        self.to_img_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.img_patch_dim, self.dim),
        )

        self.to_double_img_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.double_img_patch_dim, self.dim),
        )

        self.to_binary_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.binary_patch_dim, self.dim),
        )

        self.to_word_embedding = nn.Sequential(
            nn.Linear(300, self.dim),
        )

        # position embedding of img patch
        self.img_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        # position embedding of template patch
        self.template_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        # position embedding of the sentence
        self.ref_pos_embedding = nn.Parameter(torch.randn(1, 100, self.dim))
        # type embedding
        self.type_embedding = nn.Parameter(torch.rand(1, 1, 5))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.seg_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 1),
            nn.Sigmoid(),
            Rearrange('b (h w) c -> b c h w', h=self.num_patches_side, w=self.num_patches_side),
            #nn.Upsample(scale_factor=8, mode="bilinear")
        )

        self.class_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            Rearrange('b (h w) c -> b c h w', h=self.num_patches_side, w=self.num_patches_side),
            nn.AvgPool2d(self.num_patches_side),
            Rearrange('b c h w -> (b h w) c', h=1, w=1),
            nn.Linear(self.dim, num_classes_cls),
        )

        '''
        self.track_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            Rearrange('b (h w) c -> b c h w', h=self.num_patches_side, w=self.num_patches_side),
            nn.AvgPool2d(self.num_patches_side),
            Rearrange('b c h w -> (b h w) c', h=1, w=1),
            nn.Linear(self.dim, 4),
            nn.Sigmoid()
        )
        '''

    def forward(self, patch_dic, head):
        # embed the patches of the original image
        img_patches = patch_dic['i1']
        x = self.to_img_patch_embedding(img_patches)
        x += self.img_pos_embedding
        type_embedding_0 = repeat(self.type_embedding[:, :, 0:1], 'b () () -> b n d', n=self.num_patches, d=self.dim)
        x += type_embedding_0

        if 'double_i' in patch_dic:
            double_img_patches = patch_dic['double_i']
            x_double_img_patches = self.to_double_img_patch_embedding(double_img_patches)
            x_double_img_patches += self.img_pos_embedding
            type_embedding_0 = repeat(self.type_embedding[:, :, 1:2], 'b () () -> b n d', n=self.num_patches, d=self.dim)
            x_double_img_patches += type_embedding_0
            x = torch.cat((x, x_double_img_patches), dim=1)

        if 'i2' in patch_dic:
            img_patches_2 = patch_dic['i2']
            x_img_patches_2 = self.to_img_patch_embedding(img_patches_2)
            x_img_patches_2 += self.img_pos_embedding
            type_embedding_0 = repeat(self.type_embedding[:, :, 1:2], 'b () () -> b n d', n=self.num_patches, d=self.dim)
            x_img_patches_2 += type_embedding_0
            x = torch.cat((x, x_img_patches_2), dim=1)

        # embed the patches of template
        if 't1' in patch_dic:
            template_pathces_1 = patch_dic['t1']
            template_x_1 = self.to_binary_patch_embedding(template_pathces_1)
            template_x_1 += self.template_pos_embedding
            type_embedding_1 = repeat(self.type_embedding[:, :, 1:2], 'b () () -> b n d', n=self.num_patches, d=self.dim)
            template_x_1 += type_embedding_1
            x = torch.cat((x, template_x_1), dim=1)

        if 't1' in patch_dic and 't2' in patch_dic:
            template_pathces_2 = patch_dic['t2']
            template_x_2 = self.to_binary_patch_embedding(template_pathces_2)
            template_x_2 += self.template_pos_embedding
            type_embedding_2 = repeat(self.type_embedding[:, :, 1:2], 'b () () -> b n d', n=self.num_patches, d=self.dim)
            template_x_2 += type_embedding_2
            x = torch.cat((x, template_x_2), dim=1)

        if 'r1' in patch_dic:
            template_ref = patch_dic['r1']
            template_ref = self.to_word_embedding(template_ref)
            word_num = template_ref.shape[1]
            template_ref += self.ref_pos_embedding[:, 0:word_num, :]
            type_embedding_3 = repeat(self.type_embedding[:, :, 2:3], 'b () () -> b n d', n=word_num, d=self.dim)
            template_ref += type_embedding_3
            x = torch.cat((x, template_ref), dim=1)

        b, n, _ = x.shape
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        x = x[:,0:self.num_patches,:]

        if head == 'seg':
            x = self.seg_head(x)

        if head == 'class':
            x = self.class_head(x)

        if head == 'track':
            x = self.track_head(x)

        return x


