import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ViT & CrossViT
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# ViT & CrossViT
# FeedForward class for a feed forward neural network consisting of 2 linear layers, 
# where the first has a GELU activation followed by dropout, then the next linear layer
# followed again by dropout
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ViT & CrossViT
# Attention class for multi-head self-attention mechanism with softmax and dropout
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        # set heads and scale (=sqrt(dim_head))
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # 1/sqrt(dim_head)

        # we need softmax layer and dropout
        self.attend  = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # as well as the q linear layer
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        # and the k/v linear layer (can be realized as one single linear layer
        # or as two individual ones)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)  # Combined KV projection
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # and the output linear layer followed by dropout
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        # return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        return x.view(batch_size, seq_length, self.heads, d_model // self.heads).transpose(1, 2)


    def forward(self, x, context = None, kv_include_self = False):
        # now compute the attention/cross-attention
        # in cross attention: x = class token, context = token embeddings
        # don't forget the dropout after the attention 
        # and before the multiplication w. 'v'
        # the output should be in the shape 'b n (h d)'
        b, n, _, h = *x.shape, self.heads
        if context is None:
            context = x

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim = 1)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Method 1
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # Method 1
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        # Attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Attention weights
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Merge heads back together
        # Method 1
        # out = rearrange(out, 'b h n d -> b n (h d)')

        # Method 2
        out = out.transpose(1, 2).contiguous().view(b, -1, h * (v.size(-1)))

        # Final projection
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)



        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

# ViT & CrossViT
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

            # Self-attention with residual connection
            x = attn(x) + x # [PreNorm->Attention] + residual

            # Feed-forward with residual connection
            x = ff(x) + x # [PreNorm->FFN] + residual

        return x

# CrossViT
# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
class ProjectInOut(nn.Module):
    """
    Adapter class that embeds a callable (layer) and handles mismatching dimensions
    """
    def __init__(self, dim_outer, dim_inner, fn):
        """
        Args:
            dim_outer (int): Input (and output) dimension.
            dim_inner (int): Intermediate dimension (expected by fn).
            fn (callable): A callable object (like a layer).
        """
        super().__init__()
        self.fn = fn
        need_projection = dim_outer != dim_inner
        self.project_in = nn.Linear(dim_outer, dim_inner) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_inner, dim_outer) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        """
        Args:
            *args, **kwargs: to be passed on to fn

        Notes:
            - after calling fn, the tensor has to be projected back into it's original shape   
            - fn(W_in) * W_out
        """

        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)

        return x

# CrossViT
# cross attention transformer
class CrossTransformer(nn.Module):
    # This is a special transformer block
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Small branch: (sm_cls attends to lg_patches)
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, CrossAttention(lg_dim, heads=heads, dim_head=dim_head,
                                                               dropout=dropout))),
                # Large branch: (lg_cls attends to sm_patches)
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, CrossAttention(sm_dim, heads=heads, dim_head=dim_head,
                                                               dropout=dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        # Forward pass through the layers,
        for sm_attend_lg, lg_attend_sm in self.layers:
            # cross attend to
            # 1. small cls token to large patches and
            sm_cls = sm_attend_lg(sm_cls, lg_tokens, kv_include_self = True) + sm_cls

            # 2. large cls token to small patches
            lg_cls = lg_attend_sm(lg_cls, lg_tokens, kv_include_self = True) + lg_cls

        # finally concat sm/lg cls tokens with patch tokens 
        sm_tokens = torch.cat([sm_cls, sm_patch_tokens], dim=1)
        lg_tokens = torch.cat([lg_cls, lg_patch_tokens], dim=1)

        return sm_tokens, lg_tokens

# CrossViT
# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        # Unpack encoder parameters
        # sm_depth, sm_dim_head, sm_heads, sm_mlp_dim = sm_enc_params
        # lg_depth, lg_dim_head, lg_heads, lg_mlp_dim = lg_enc_params

        sm_depth = sm_enc_params['depth']
        sm_heads = sm_enc_params['heads']
        sm_dim_head = sm_enc_params['dim_head']
        sm_mlp_dim = sm_enc_params['mlp_dim']

        lg_depth = lg_enc_params['depth']
        lg_heads = lg_enc_params['heads']
        lg_dim_head = lg_enc_params['dim_head']
        lg_mlp_dim = lg_enc_params['mlp_dim']

        for _ in range(depth):
            sm_encoder = Transformer(dim=sm_dim, depth=sm_depth, heads=sm_heads, dim_head=sm_dim_head, mlp_dim=sm_mlp_dim, dropout=dropout)
            lg_encoder = Transformer(dim=lg_dim, depth=lg_depth, heads=lg_heads, dim_head=lg_dim_head, mlp_dim=lg_mlp_dim, dropout=dropout)
            cross_transformer = CrossTransformer(sm_dim, lg_dim, cross_attn_depth, cross_attn_heads, cross_attn_dim_head, dropout=dropout)

            self.layers.append(nn.ModuleList([
                # 2 transformer branches, one for small, one for large patchs
                # + 1 cross transformer block
                sm_encoder,
                lg_encoder,
                cross_transformer
            ]))

    def forward(self, sm_tokens, lg_tokens):
        # forward through the transformer encoders and cross attention block

        for sm_enc, lg_enc, cross_attn in self.layers:
            sm_tokens = sm_tokens + sm_enc(sm_tokens)
            lg_tokens = lg_tokens + lg_enc(lg_tokens)

            sm_tokens, lg_tokens = cross_attn(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# CrossViT (could actually also be used in ViT)
# helper function that makes the embedding from patches
# have a look at the image embedding in ViT
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # create layer that re-arranges the image patches
        # and embeds them with layer norm + linear projection + layer norm
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # create/initialize #dim-dimensional positional embedding (will be learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # create #dim cls tokens (for each patch embedding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # create dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        # Convert image to patch embeddings
        x = self.to_patch_embedding(img) # Shape: [batch, num_patches, dim]
        b, n, _ = x.shape  # batch size, number of patches, dimension

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) #'1 1 d -> b 1 d', b=b)


        # forward through patch embedding layer
        # concat class tokens
        x = torch.cat((cls_tokens, x), dim=1)  # shape: (b, num_patches + 1, dim)
        # and add positional embedding
        x += self.pos_embedding[:, :(n + 1)]  # Using n+1 to match new length
        return self.dropout(x)


# normal ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # initialize patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # create transformer blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # create mlp head (layer norm + linear layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # concat class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        # apply dropout
        x = self.dropout(x)

        # forward through the transformer blocks
        x = self.transformer(x)

        # decide if x is the mean of the embedding 
        # or the class token
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # transfer via an identity layer the cls tokens or the mean
        # to a latent space, which can then be used as input
        # to the mlp head
        x = self.to_latent(x)
        return self.mlp_head(x)


# CrossViT
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        # create ImageEmbedder for small and large patches
        # TODO
        self.sm_image_embedder = ImageEmbedder(dim=sm_dim, image_size=image_size, patch_size=sm_patch_size, dropout=dropout)
        # self.sm_head = nn.Sequential()
        self.lg_image_embedder = ImageEmbedder(dim=lg_dim, image_size=image_size, patch_size=lg_patch_size, dropout=dropout)


        # create MultiScaleEncoder
        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        # create mlp heads for small and large patches
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        # apply image embedders
        ## Get embeddings for both scales
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        # and the multi-scale encoder
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        # call the mlp heads w. the class tokens 
        # Get predictions from both branches
        sm_logits = self.sm_mlp_head(sm_tokens)
        lg_logits = self.lg_mlp_head(lg_tokens)
        
        return sm_logits + lg_logits


if __name__ == "__main__":
    x = torch.randn(16, 3, 32, 32)
    # vit = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64, depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1, emb_dropout = 0.1)
    cvit = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, lg_dim = 128, sm_patch_size = 8,
                    sm_enc_depth = 2, sm_enc_heads = 8, sm_enc_mlp_dim = 128, sm_enc_dim_head = 64,
                    lg_patch_size = 16, lg_enc_depth = 2, lg_enc_heads = 8, lg_enc_mlp_dim = 128,
                    lg_enc_dim_head = 64, cross_attn_depth = 2, cross_attn_heads = 8, cross_attn_dim_head = 64,
                    depth = 3, dropout = 0.1, emb_dropout = 0.1)
    # print(vit(x).shape)
    print(cvit(x).shape)
