import torch
import torch.nn as nn
import torch.nn.functional as F
from damg_network.feature_transformer import *

try:
    from einops import rearrange
except ImportError:
    def rearrange(x, pattern, **kwargs):
        if pattern == 'b (head c) h w -> b head c (h w)':
            head = kwargs['head']
            b, hc, h, w = x.shape
            c = hc // head
            return x.view(b, head, c, h * w)
        if pattern == 'b head c (h w) -> b (head c) h w':
            head = kwargs['head']
            h = kwargs['h']
            w = kwargs['w']
            b, _, c, _ = x.shape
            return x.view(b, head * c, h, w)
        raise NotImplementedError(f'Unsupported rearrange pattern without einops: {pattern}')

# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

class EvidenceRouter(nn.Module):
    def __init__(self, hvi_bias=1.0, sem_bias=1.0, geo_bias=1.0):
        super().__init__()
        self.register_buffer(
            'branch_bias',
            torch.tensor([hvi_bias, sem_bias, geo_bias], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, obs, rs, rg, reference, sem_valid=None, geo_valid=None):
        b, _, h, w = reference.shape
        if obs is None:
            obs = torch.ones((b, 1, h, w), device=reference.device, dtype=reference.dtype)
        elif obs.shape[-2:] != (h, w):
            obs = F.interpolate(obs, size=(h, w), mode='bilinear', align_corners=False)

        if rs is None:
            rs = torch.zeros_like(obs)
        elif rs.shape[-2:] != (h, w):
            rs = F.interpolate(rs, size=(h, w), mode='bilinear', align_corners=False)

        if rg is None:
            rg = torch.zeros_like(obs)
        elif rg.shape[-2:] != (h, w):
            rg = F.interpolate(rg, size=(h, w), mode='bilinear', align_corners=False)

        if sem_valid is None:
            sem_valid = torch.ones_like(obs)
        elif sem_valid.shape[-2:] != (h, w):
            sem_valid = F.interpolate(sem_valid, size=(h, w), mode='bilinear', align_corners=False)

        if geo_valid is None:
            geo_valid = torch.ones_like(obs)
        elif geo_valid.shape[-2:] != (h, w):
            geo_valid = F.interpolate(geo_valid, size=(h, w), mode='bilinear', align_corners=False)

        hvi_gate = obs
        sem_gate = (1.0 - obs) * rs * sem_valid
        geo_gate = (1.0 - obs) * rg * geo_valid
        raw_weights = torch.cat([hvi_gate, sem_gate, geo_gate], dim=1).clamp_min(0.0)
        branch_bias = self.branch_bias.to(device=reference.device, dtype=reference.dtype)
        raw_weights = raw_weights * branch_bias
        weight_sum = raw_weights.sum(dim=1, keepdim=True)
        route = raw_weights / (weight_sum + 1e-6)
        conservative_strength = torch.clamp(weight_sum, 0.0, 1.0)
        return route, conservative_strength
    

# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
  
  
# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim) # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        self.ffn_sem = CAB(dim, num_heads, bias)
        self.ffn_geo = CAB(dim, num_heads, bias)
        self.router = EvidenceRouter(hvi_bias=1.00, sem_bias=1.18, geo_bias=0.82)
        
    def forward(self, x, y, sem=None, geo=None, obs=None, rs=None, rg=None, sem_valid=None, geo_valid=None, return_route=False):
        norm_x = self.norm(x)
        norm_y = self.norm(y)
        norm_sem = norm_y if sem is None else self.norm(sem)
        norm_geo = norm_y if geo is None else self.norm(geo)

        route, strength = self.router(obs, rs, rg, x, sem_valid=sem_valid, geo_valid=geo_valid)
        fused = (
            route[:, 0:1] * self.ffn(norm_x, norm_y) +
            route[:, 1:2] * self.ffn_sem(norm_x, norm_sem) +
            route[:, 2:3] * self.ffn_geo(norm_x, norm_geo)
        )
        x = x + (strength * fused)
        x = self.gdfn(self.norm(x))
        if return_route:
            return x, route, strength
        return x
    
class I_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.ffn_sem = CAB(dim, num_heads, bias=bias)
        self.ffn_geo = CAB(dim, num_heads, bias=bias)
        self.router = EvidenceRouter(hvi_bias=1.00, sem_bias=0.90, geo_bias=1.20)
        
    def forward(self, x, y, sem=None, geo=None, obs=None, rs=None, rg=None, sem_valid=None, geo_valid=None, return_route=False):
        norm_x = self.norm(x)
        norm_y = self.norm(y)
        norm_sem = norm_y if sem is None else self.norm(sem)
        norm_geo = norm_y if geo is None else self.norm(geo)

        route, strength = self.router(obs, rs, rg, x, sem_valid=sem_valid, geo_valid=geo_valid)
        fused = (
            route[:, 0:1] * self.ffn(norm_x, norm_y) +
            route[:, 1:2] * self.ffn_sem(norm_x, norm_sem) +
            route[:, 2:3] * self.ffn_geo(norm_x, norm_geo)
        )
        x = x + (strength * fused)
        x = x + self.gdfn(self.norm(x))
        if return_route:
            return x, route, strength
        return x
