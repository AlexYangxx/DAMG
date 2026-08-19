import torch
import torch.nn.functional as F


def _gradient_map(x):
    grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
    return grad_x + grad_y


def _pseudo_normal(depth):
    grad_x = F.pad(depth[:, :, :, 1:] - depth[:, :, :, :-1], (0, 1, 0, 0), mode='replicate')
    grad_y = F.pad(depth[:, :, 1:, :] - depth[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')
    normal = torch.cat([-grad_x, -grad_y, torch.ones_like(depth)], dim=1)
    return F.normalize(normal, dim=1)


def build_placeholder_priors(rgb):
    depth = F.avg_pool2d(rgb.mean(dim=1, keepdim=True), kernel_size=5, stride=1, padding=2)
    semantic = rgb
    normal = _pseudo_normal(depth)
    geometry = torch.cat([depth, normal], dim=1)
    confidence_hint = torch.clamp(
        0.6 * depth + 0.4 * torch.clamp(4.0 * _gradient_map(depth), 0.0, 1.0),
        0.0,
        1.0,
    )
    return {
        'semantic': semantic,
        'depth': depth,
        'normal': normal,
        'geometry': geometry,
        'confidence_hint': confidence_hint,
    }
