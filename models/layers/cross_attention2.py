from math import ceil

import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, reduce

import pdb
import torch.nn.functional as F
"""

Contains the custom implementation of cross attention between pathways and histology and self attention between pathways 

"""

NUM_PATHWAYS = 1280

def exists(val):
    return val is not None


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 128,    # number of landmarks
        pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        num_pathways = 281,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) #256, 128*3

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, eps, iters = *x.shape, self.heads, self.num_landmarks, self.eps, self.pinv_iterations

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :self.num_pathways, :]

        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, self.num_pathways:, :]
        
        # ensure `l` is calculated based on each part's length
        l_pathways = ceil(self.num_pathways / m)  # calculate landmarks for pathways
        l_histology = ceil((n - self.num_pathways) / m)  # calculate landmarks for histology

        # pad q_pathways and q_histology if necessary for landmarks calculation
        remainder_pathways = self.num_pathways % l_pathways
        if remainder_pathways > 0:
            padding_pathways = l_pathways - remainder_pathways
            q_pathways = F.pad(q_pathways, (0, 0, padding_pathways, 0), value=0)
            k_pathways = F.pad(k_pathways, (0, 0, padding_pathways, 0), value=0)

        remainder_histology = (n - self.num_pathways) % l_histology
        if remainder_histology > 0:
            padding_histology = l_histology - remainder_histology
            q_histology = F.pad(q_histology, (0, 0, 0, padding_histology), value=0)
            k_histology = F.pad(k_histology, (0, 0, 0, padding_histology), value=0)

        # landmarks for both pathways and histology
        q_landmarks_pathways = reduce(q_pathways, '... (n l) d -> ... n d', 'sum', l=l_pathways)
        k_landmarks_pathways = reduce(k_pathways, '... (n l) d -> ... n d', 'sum', l=l_pathways)

        q_landmarks_histology = reduce(q_histology, '... (n l) d -> ... n d', 'sum', l=l_histology)
        k_landmarks_histology = reduce(k_histology, '... (n l) d -> ... n d', 'sum', l=l_histology)

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        # cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        # attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        # cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        sim1 = einsum(einops_eq, q_histology, k_landmarks_pathways)  # histology to pathways landmarks
        sim2 = einsum(einops_eq, q_landmarks_pathways, k_landmarks_pathways)  # pathways landmarks to pathways landmarks
        sim3 = einsum(einops_eq, q_landmarks_pathways, k_histology)  # pathways landmarks to histology

        # softmax
        # pre_softmax_cross_attn_histology = cross_attn_histology
        # cross_attn_histology = cross_attn_histology.softmax(dim=-1)
        # attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

        # compute output 
        # out_pathways =  attn_pathways_histology @ v
        # out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]

        # attention weights using softmax
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)  # calculate the pseudo-inverse for attn2

        # compute output for pathways and histology
        out_histology = (attn1 @ attn2_inv) @ (attn3 @ v[:, :, :self.num_pathways])
        out_pathways = attn2 @ v

        out = torch.cat((out_pathways, out_histology), dim=2)
        
        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        if return_attn:  
            # return three matrices
            # return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()
            return out, attn1.detach().cpu(), attn2.detach().cpu(), attn3.detach().cpu()


        return out

class MMAttention2(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 128,    # number of landmarks
        pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        num_pathways = 281,
    ):
        super().__init__()
        self.num_pathways = 4427 #num_pathways
        self.eps = eps
        inner_dim = heads * dim_head #1 * 64

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) #256, 128*3

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask=None, return_attn=False, num_wsi1pathway=None):
        b, n, _, h, m, eps, iters = *x.shape, self.heads, self.num_landmarks, self.eps, self.pinv_iterations

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :num_wsi1pathway, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :num_wsi1pathway, :]

        q_histology = q[:, :, num_wsi1pathway:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, num_wsi1pathway:, :]
        
        # ensure `l` is calculated based on each part's length
        l_pathways = ceil(num_wsi1pathway / m)  # calculate landmarks for pathways
        l_histology = ceil((n - num_wsi1pathway) / m)  # calculate landmarks for histology


        # landmarks generation for both pathways and histology
        l = ceil(n / m)  # calculate landmarks
        q_landmarks = reduce(q, '... (n l) d -> ... n d', 'sum', l=l)
        k_landmarks = reduce(k, '... (n l) d -> ... n d', 'sum', l=l)
        
        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        # cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        # attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        # cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        sim1 = einsum(einops_eq, q_histology, k_landmarks)  # histology to pathways landmarks
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)  # landmarks to landmarks
        sim3 = einsum(einops_eq, q_landmarks, k_histology)  # pathways landmarks to histology

        # softmax
        # pre_softmax_cross_attn_histology = cross_attn_histology
        # cross_attn_histology = cross_attn_histology.softmax(dim=-1)
        # attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

        # compute output 
        # out_pathways =  attn_pathways_histology @ v
        # out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]
        # out_histology = cross_attn_histology @ v[:, :, :num_wsi1pathway]

        # attention weights using softmax
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)  # calculate the pseudo-inverse for attn2

        # compute output for pathways and histology
        out_histology = (attn1 @ attn2_inv) @ (attn3 @ v[:, :, :self.num_pathways])
        out_pathways = attn2 @ v

        out = torch.cat((out_pathways, out_histology), dim=2)
        
        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        if return_attn:  
            # return three matrices
            # return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()
            return out, attn1.detach().cpu(), attn2.detach().cpu(), attn3.detach().cpu()

        return out

class MMAttention3(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 128,    # number of landmarks
        pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        num_pathways = 281,
    ):
        super().__init__()
        self.num_pathways = 8523 #num_pathways
        self.eps = eps
        inner_dim = heads * dim_head #1 * 32

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) #256, 128*3

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask=None, return_attn=False, num_wsi1pathway=None):
        b, n, _, h, m, eps, iters = *x.shape, self.heads, self.num_pathways, self.eps, self.pinv_iterations

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :num_wsi1pathway, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :num_wsi1pathway, :]

        q_histology = q[:, :, num_wsi1pathway:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, num_wsi1pathway:, :]
        
        # landmarks generation for both pathways and histology
        l = ceil(n / m)  # calculate landmarks
        q_landmarks = reduce(q, '... (n l) d -> ... n d', 'sum', l=l)
        k_landmarks = reduce(k, '... (n l) d -> ... n d', 'sum', l=l)
        
        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        # cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        # attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        # cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        sim1 = einsum(einops_eq, q_histology, k_landmarks)  # histology to pathways landmarks
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)  # landmarks to landmarks
        sim3 = einsum(einops_eq, q_landmarks, k_histology)  # pathways landmarks to histology

        # softmax
        # pre_softmax_cross_attn_histology = cross_attn_histology
        # cross_attn_histology = cross_attn_histology.softmax(dim=-1)
        # attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

        # compute output 
        # out_pathways =  attn_pathways_histology @ v
        # out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]
        # out_histology = cross_attn_histology @ v[:, :, :num_wsi1pathway]

        # attention weights using softmax
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)  # calculate the pseudo-inverse for attn2

        # compute output for pathways and histology
        out_histology = (attn1 @ attn2_inv) @ (attn3 @ v[:, :, :self.num_pathways])
        out_pathways = attn2 @ v

        out = torch.cat((out_pathways, out_histology), dim=2)
        
        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        if return_attn:  
            # return three matrices
            # return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()
            return out, attn1.detach().cpu(), attn2.detach().cpu(), attn3.detach().cpu()

        return out
        
class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        num_landmarks = 128,    # number of landmarks
        pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        residual=True,
        dropout=0.,
        num_pathways = 281,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways #331
        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_landmarks = 128,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = x + self.attn(x=self.norm(x), mask=mask)

        return x

class MMAttentionLayer2(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        num_landmarks = 128,    # number of landmarks
        pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        residual=True,
        dropout=0.,
        num_pathways = 281,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = MMAttention2(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_landmarks = 128,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )

    def forward(self, x=None, mask=None, return_attention=False, num_wsi1pathway=None):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask, return_attn=True, num_wsi1pathway=num_wsi1pathway)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = x + self.attn(x=self.norm(x), mask=mask, num_wsi1pathway=num_wsi1pathway)

        return x

class MMAttentionLayer3(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        num_landmarks = 128,    # number of landmarks
        pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        residual=True,
        dropout=0.,
        num_pathways = 281,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = MMAttention3(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_landmarks = 128,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = x + self.attn(x=self.norm(x), mask=mask)

        return x