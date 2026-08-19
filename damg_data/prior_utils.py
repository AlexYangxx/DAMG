import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PRIOR_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.npy', '.npz', '.pt', '.pth')


@dataclass
class PriorConfig:
    enabled: bool = False
    strict: bool = False
    semantic_dir: str = ''
    depth_dir: str = ''
    normal_dir: str = ''
    semantic_quality_dir: str = ''
    depth_quality_dir: str = ''
    normal_quality_dir: str = ''
    semantic_index: dict = field(default_factory=dict)
    depth_index: dict = field(default_factory=dict)
    normal_index: dict = field(default_factory=dict)
    semantic_quality_index: dict = field(default_factory=dict)
    depth_quality_index: dict = field(default_factory=dict)
    normal_quality_index: dict = field(default_factory=dict)


def _normalize_key(value):
    return value.replace('\\', '/').strip('/').lower()


def _remove_extension(value):
    return os.path.splitext(value)[0]


def _index_prior_root(root):
    index = {'root': root, 'by_rel': {}, 'by_stem': {}}
    if not root or not os.path.isdir(root):
        return index

    for current_root, _, files in os.walk(root):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in PRIOR_EXTENSIONS:
                continue
            full_path = os.path.join(current_root, file_name)
            rel_path = _normalize_key(os.path.relpath(full_path, root))
            rel_key = _remove_extension(rel_path)
            stem_key = os.path.splitext(os.path.basename(file_name))[0].lower()
            index['by_rel'][rel_key] = full_path
            index['by_stem'].setdefault(stem_key, []).append(full_path)
    return index


def build_prior_config(
    semantic_dir='',
    depth_dir='',
    normal_dir='',
    semantic_quality_dir='',
    depth_quality_dir='',
    normal_quality_dir='',
    strict=False,
):
    enabled = any([
        semantic_dir and os.path.isdir(semantic_dir),
        depth_dir and os.path.isdir(depth_dir),
        normal_dir and os.path.isdir(normal_dir),
        semantic_quality_dir and os.path.isdir(semantic_quality_dir),
        depth_quality_dir and os.path.isdir(depth_quality_dir),
        normal_quality_dir and os.path.isdir(normal_quality_dir),
    ])
    return PriorConfig(
        enabled=enabled,
        strict=strict,
        semantic_dir=semantic_dir,
        depth_dir=depth_dir,
        normal_dir=normal_dir,
        semantic_quality_dir=semantic_quality_dir,
        depth_quality_dir=depth_quality_dir,
        normal_quality_dir=normal_quality_dir,
        semantic_index=_index_prior_root(semantic_dir),
        depth_index=_index_prior_root(depth_dir),
        normal_index=_index_prior_root(normal_dir),
        semantic_quality_index=_index_prior_root(semantic_quality_dir),
        depth_quality_index=_index_prior_root(depth_quality_dir),
        normal_quality_index=_index_prior_root(normal_quality_dir),
    )


def _candidate_relative_keys(sample_path, reference_root=None):
    if sample_path is None:
        return []

    keys = []
    sample_path = os.path.normpath(sample_path)
    if reference_root and os.path.exists(sample_path):
        try:
            rel_path = _normalize_key(os.path.relpath(sample_path, reference_root))
            rel_key = _remove_extension(rel_path)
            keys.append(rel_key)
            if '/' in rel_key:
                keys.append('/'.join(rel_key.split('/')[1:]))
        except ValueError:
            pass

    sample_norm = _normalize_key(sample_path)
    keys.append(_remove_extension(sample_norm))
    keys.append(os.path.splitext(os.path.basename(sample_path))[0].lower())

    unique = []
    for key in keys:
        if key and key not in unique:
            unique.append(key)
    return unique


def resolve_prior_path(index, sample_path, reference_root=None):
    if not index or not index.get('root'):
        return None

    for key in _candidate_relative_keys(sample_path, reference_root=reference_root):
        if key in index['by_rel']:
            return index['by_rel'][key]
        stem_matches = index['by_stem'].get(key)
        if stem_matches:
            if len(stem_matches) == 1:
                return stem_matches[0]
            exact_name = os.path.splitext(os.path.basename(sample_path))[0].lower()
            for candidate in stem_matches:
                if os.path.splitext(os.path.basename(candidate))[0].lower() == exact_name:
                    return candidate
    return None


def _ensure_chw(tensor):
    while tensor.dim() > 3:
        tensor = tensor[0]

    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(f'Unsupported prior tensor shape: {tuple(tensor.shape)}')

    c0, c1, c2 = tensor.shape

    if c0 in (1, 3, 4):
        return tensor
    if c2 in (1, 3, 4):
        return tensor.permute(2, 0, 1)

    if c0 <= 16 and c1 > 16 and c2 > 16:
        return tensor
    if c2 <= 16 and c0 > 16 and c1 > 16:
        return tensor.permute(2, 0, 1)

    if c0 >= max(c1, c2):
        return tensor
    if c2 >= max(c0, c1):
        return tensor.permute(2, 0, 1)

    smallest_dim = min(range(3), key=lambda idx: tensor.shape[idx])
    if smallest_dim == 0:
        return tensor
    if smallest_dim == 2:
        return tensor.permute(2, 0, 1)
    return tensor.permute(1, 0, 2)


def _load_npz_first_array(path):
    npz_file = np.load(path)
    if isinstance(npz_file, np.ndarray):
        return npz_file
    preferred_keys = ['arr_0', 'feat', 'feature', 'features', 'embedding', 'tensor', 'depth', 'normal']
    for key in preferred_keys:
        if key in npz_file:
            return npz_file[key]
    first_key = next(iter(npz_file.keys()))
    return npz_file[first_key]


def load_prior_tensor(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        array = np.array(Image.open(path), copy=True)
        if array.ndim == 2:
            array = array[..., None]
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        return tensor

    if ext == '.npy':
        array = np.load(path)
        tensor = torch.from_numpy(array).float()
        return _ensure_chw(tensor)

    if ext == '.npz':
        array = _load_npz_first_array(path)
        tensor = torch.from_numpy(np.asarray(array)).float()
        return _ensure_chw(tensor)

    if ext in ('.pt', '.pth'):
        obj = torch.load(path, map_location='cpu')
        if torch.is_tensor(obj):
            return _ensure_chw(obj.float())
        if isinstance(obj, dict):
            preferred_keys = ['feat', 'feature', 'features', 'embedding', 'tensor', 'depth', 'normal']
            for key in preferred_keys:
                if key in obj and torch.is_tensor(obj[key]):
                    return _ensure_chw(obj[key].float())
            for value in obj.values():
                if torch.is_tensor(value):
                    return _ensure_chw(value.float())
        raise ValueError(f'Unsupported tensor container in {path}')

    raise ValueError(f'Unsupported prior extension: {ext}')


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


def _pil_to_tensor(image):
    array = np.array(image, copy=True)
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def _resize_tensor_like(tensor, size):
    if tensor.shape[-2:] == size:
        return tensor
    return F.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)


def _pad_if_smaller(tensor, target_h, target_w):
    h, w = tensor.shape[-2:]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return tensor

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')


def _crop_tensor(tensor, top, left, crop_size):
    return tensor[:, top:top + crop_size, left:left + crop_size]


def _maybe_flip(tensor, horizontal=False, vertical=False):
    if horizontal:
        tensor = torch.flip(tensor, dims=[2])
    if vertical:
        tensor = torch.flip(tensor, dims=[1])
    return tensor


def load_real_priors(low_tensor, sample_path, prior_config=None, reference_root=None):
    low_h, low_w = low_tensor.shape[-2:]
    depth_default = low_tensor.mean(dim=0, keepdim=True)

    semantic_path = None
    depth_path = None
    normal_path = None
    semantic_quality_path = None
    depth_quality_path = None
    normal_quality_path = None
    semantic = low_tensor.clone()
    depth = depth_default.clone()
    semantic_valid = torch.ones(1, low_h, low_w)
    depth_valid = torch.ones(1, low_h, low_w)
    normal_valid = torch.ones(1, low_h, low_w)

    if prior_config and prior_config.enabled:
        semantic_path = resolve_prior_path(prior_config.semantic_index, sample_path, reference_root=reference_root)
        depth_path = resolve_prior_path(prior_config.depth_index, sample_path, reference_root=reference_root)
        normal_path = resolve_prior_path(prior_config.normal_index, sample_path, reference_root=reference_root)
        semantic_quality_path = resolve_prior_path(prior_config.semantic_quality_index, sample_path, reference_root=reference_root)
        depth_quality_path = resolve_prior_path(prior_config.depth_quality_index, sample_path, reference_root=reference_root)
        normal_quality_path = resolve_prior_path(prior_config.normal_quality_index, sample_path, reference_root=reference_root)

        if semantic_path:
            semantic = _resize_tensor_like(load_prior_tensor(semantic_path), (low_h, low_w))
        else:
            semantic_valid = torch.zeros_like(semantic_valid)

        if depth_path:
            depth = _resize_tensor_like(load_prior_tensor(depth_path), (low_h, low_w))
            if depth.shape[0] > 1:
                depth = depth[:1]
        else:
            depth_valid = torch.zeros_like(depth_valid)

        if normal_path:
            normal = _resize_tensor_like(load_prior_tensor(normal_path), (low_h, low_w))
            if normal.shape[0] == 1:
                normal = normal.repeat(3, 1, 1)
            elif normal.shape[0] > 3:
                normal = normal[:3]
        else:
            normal_valid = torch.zeros_like(normal_valid)
            normal = _pseudo_normal(depth.unsqueeze(0)).squeeze(0)

        if prior_config.strict:
            if prior_config.semantic_dir and semantic_path is None:
                raise FileNotFoundError(f'Missing semantic prior for sample: {sample_path}')
            if prior_config.depth_dir and depth_path is None:
                raise FileNotFoundError(f'Missing depth prior for sample: {sample_path}')
            if prior_config.normal_dir and normal_path is None:
                raise FileNotFoundError(f'Missing normal prior for sample: {sample_path}')
            if prior_config.semantic_quality_dir and semantic_quality_path is None:
                raise FileNotFoundError(f'Missing semantic quality prior for sample: {sample_path}')
            if prior_config.depth_quality_dir and depth_quality_path is None:
                raise FileNotFoundError(f'Missing depth quality prior for sample: {sample_path}')
            if prior_config.normal_quality_dir and normal_quality_path is None:
                raise FileNotFoundError(f'Missing normal quality prior for sample: {sample_path}')
    else:
        normal = _pseudo_normal(depth.unsqueeze(0)).squeeze(0)

    if semantic_quality_path:
        semantic_quality = _resize_tensor_like(load_prior_tensor(semantic_quality_path), (low_h, low_w))
        if semantic_quality.shape[0] > 1:
            semantic_quality = semantic_quality[:1]
        semantic_quality = torch.clamp(semantic_quality, 0.0, 1.0)
    else:
        semantic_quality = semantic_valid.clone()

    if depth_quality_path:
        depth_quality = _resize_tensor_like(load_prior_tensor(depth_quality_path), (low_h, low_w))
        if depth_quality.shape[0] > 1:
            depth_quality = depth_quality[:1]
        depth_quality = torch.clamp(depth_quality, 0.0, 1.0)
    else:
        depth_quality = depth_valid.clone()

    if normal_quality_path:
        normal_quality = _resize_tensor_like(load_prior_tensor(normal_quality_path), (low_h, low_w))
        if normal_quality.shape[0] > 1:
            normal_quality = normal_quality[:1]
        normal_quality = torch.clamp(normal_quality, 0.0, 1.0)
    else:
        normal_quality = normal_valid.clone()

    geometry = torch.cat([depth, normal], dim=0)
    return {
        'semantic': semantic,
        'depth': depth,
        'normal': normal,
        'geometry': geometry,
        'semantic_quality': semantic_quality,
        'depth_quality': depth_quality,
        'normal_quality': normal_quality,
        'semantic_valid': semantic_valid,
        'depth_valid': depth_valid,
        'normal_valid': normal_valid,
    }


def prepare_training_sample(low_img, high_img, sample_path, crop_size=None, prior_config=None, reference_root=None):
    low_tensor = _pil_to_tensor(low_img)
    high_tensor = _pil_to_tensor(high_img)
    priors = load_real_priors(low_tensor, sample_path, prior_config=prior_config, reference_root=reference_root)

    if high_tensor.shape[-2:] != low_tensor.shape[-2:]:
        high_tensor = _resize_tensor_like(high_tensor, low_tensor.shape[-2:])

    if crop_size is not None:
        low_tensor = _pad_if_smaller(low_tensor, crop_size, crop_size)
        high_tensor = _pad_if_smaller(high_tensor, crop_size, crop_size)
        priors = {key: _pad_if_smaller(value, crop_size, crop_size) for key, value in priors.items()}

        height, width = low_tensor.shape[-2:]
        top = 0 if height == crop_size else random.randint(0, height - crop_size)
        left = 0 if width == crop_size else random.randint(0, width - crop_size)

        low_tensor = _crop_tensor(low_tensor, top, left, crop_size)
        high_tensor = _crop_tensor(high_tensor, top, left, crop_size)
        priors = {key: _crop_tensor(value, top, left, crop_size) for key, value in priors.items()}

    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    low_tensor = _maybe_flip(low_tensor, horizontal=hflip, vertical=vflip)
    high_tensor = _maybe_flip(high_tensor, horizontal=hflip, vertical=vflip)
    priors = {key: _maybe_flip(value, horizontal=hflip, vertical=vflip) for key, value in priors.items()}

    return low_tensor, high_tensor, priors


def prepare_eval_sample(input_img, sample_path, prior_config=None, reference_root=None, pad_to_factor=None):
    input_tensor = _pil_to_tensor(input_img)
    priors = load_real_priors(input_tensor, sample_path, prior_config=prior_config, reference_root=reference_root)
    original_h, original_w = input_tensor.shape[-2:]

    if pad_to_factor is not None:
        factor = pad_to_factor
        h, w = input_tensor.shape[-2:]
        target_h = ((h + factor) // factor) * factor
        target_w = ((w + factor) // factor) * factor
        pad_h = target_h - h if h % factor != 0 else 0
        pad_w = target_w - w if w % factor != 0 else 0
        if pad_h != 0 or pad_w != 0:
            input_tensor = F.pad(input_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            priors = {
                key: F.pad(value.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
                for key, value in priors.items()
            }

    return input_tensor, priors, original_h, original_w
