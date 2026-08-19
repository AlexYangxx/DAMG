import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from tqdm import tqdm
from damg_data.data import *
from damg_data.prior_utils import build_prior_config
from torchvision import transforms
from torch.utils.data import DataLoader
from restoration_objectives.losses import *
from damg_network.damg import DAMG


def load_checkpoint_compatible(model, model_path):
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = 0
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if tuple(target.shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        filtered_state[key] = value
    incompatible = model.load_state_dict(filtered_state, strict=False)
    missing = len(getattr(incompatible, 'missing_keys', []))
    unexpected = len(getattr(incompatible, 'unexpected_keys', []))
    print(f'Pre-trained model is loaded with missing={missing}, unexpected={unexpected}, skipped_shape={skipped_shape}.')


def save_single_map(tensor, path):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3 and tensor.shape[0] > 3:
        tensor = tensor[:1]
    image = transforms.ToPILImage()(tensor)
    image.save(path)


def average_route_stack(route_stack, size):
    if not route_stack:
        return None
    merged = None
    for route in route_stack:
        resized = F.interpolate(route, size=size, mode='bilinear', align_corners=False)
        merged = resized if merged is None else merged + resized
    return merged / max(len(route_stack), 1)


def clone_priors(priors):
    if not isinstance(priors, dict):
        return priors
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in priors.items()
    }


def horizontal_flip_priors(priors):
    if not isinstance(priors, dict):
        return priors
    flipped = clone_priors(priors)
    for key, value in flipped.items():
        if torch.is_tensor(value) and value.dim() >= 4:
            flipped[key] = torch.flip(value, dims=[3])
    if 'normal' in flipped and torch.is_tensor(flipped['normal']) and flipped['normal'].shape[1] >= 1:
        flipped['normal'][:, 0:1, :, :] = -flipped['normal'][:, 0:1, :, :]
    if 'geometry' in flipped and torch.is_tensor(flipped['geometry']) and flipped['geometry'].shape[1] >= 2:
        flipped['geometry'][:, 1:2, :, :] = -flipped['geometry'][:, 1:2, :, :]
    return flipped


def vertical_flip_priors(priors):
    if not isinstance(priors, dict):
        return priors
    flipped = clone_priors(priors)
    for key, value in flipped.items():
        if torch.is_tensor(value) and value.dim() >= 4:
            flipped[key] = torch.flip(value, dims=[2])
    if 'normal' in flipped and torch.is_tensor(flipped['normal']) and flipped['normal'].shape[1] >= 2:
        flipped['normal'][:, 1:2, :, :] = -flipped['normal'][:, 1:2, :, :]
    if 'geometry' in flipped and torch.is_tensor(flipped['geometry']) and flipped['geometry'].shape[1] >= 3:
        flipped['geometry'][:, 2:3, :, :] = -flipped['geometry'][:, 2:3, :, :]
    return flipped


def flip_priors(priors, dims):
    flipped = priors
    if 3 in dims:
        flipped = horizontal_flip_priors(flipped)
    if 2 in dims:
        flipped = vertical_flip_priors(flipped)
    return flipped


def _flip_back_tensor(value, dims):
    if torch.is_tensor(value) and value.dim() >= 4:
        return torch.flip(value, dims=list(dims))
    if isinstance(value, list):
        return [_flip_back_tensor(item, dims) for item in value]
    if isinstance(value, dict):
        return {
            key: _flip_back_tensor(item, dims)
            for key, item in value.items()
        }
    return value


def _average_outputs(base, aug):
    if torch.is_tensor(base) and torch.is_tensor(aug):
        return 0.5 * (base + aug)
    if isinstance(base, list) and isinstance(aug, list):
        return [_average_outputs(b, a) for b, a in zip(base, aug)]
    if isinstance(base, dict) and isinstance(aug, dict):
        merged = {}
        for key in base.keys():
            if key in aug:
                merged[key] = _average_outputs(base[key], aug[key])
            else:
                merged[key] = base[key]
        return merged
    return base


def _sum_outputs(base, aug):
    if torch.is_tensor(base) and torch.is_tensor(aug):
        return base + aug
    if isinstance(base, list) and isinstance(aug, list):
        return [_sum_outputs(b, a) for b, a in zip(base, aug)]
    if isinstance(base, dict) and isinstance(aug, dict):
        merged = {}
        for key in base.keys():
            if key in aug:
                merged[key] = _sum_outputs(base[key], aug[key])
            else:
                merged[key] = base[key]
        return merged
    return base


def _divide_output(value, divisor):
    if torch.is_tensor(value):
        return value / divisor
    if isinstance(value, list):
        return [_divide_output(item, divisor) for item in value]
    if isinstance(value, dict):
        return {
            key: _divide_output(item, divisor)
            for key, item in value.items()
        }
    return value


def dump_auxiliary_outputs(output_folder, file_name, model_output, crop_hw=None):
    aux_root = os.path.join(output_folder, '_aux')
    os.makedirs(aux_root, exist_ok=True)

    aux_maps = {
        'obs': model_output.get('obs'),
        'rs': model_output.get('rs'),
        'rg': model_output.get('rg'),
        'semantic_quality': model_output.get('semantic_quality'),
        'depth_quality': model_output.get('depth_quality'),
        'normal_quality': model_output.get('normal_quality'),
        'geometry_quality': model_output.get('geometry_quality'),
        'prior_quality': model_output.get('prior_quality'),
        'semantic_support': model_output.get('semantic_support'),
        'geometry_support': model_output.get('geometry_support'),
        'support_consensus': model_output.get('support_consensus'),
        'prior_support_base': model_output.get('prior_support_base'),
        'prior_support_learned': model_output.get('prior_support_learned'),
        'prior_support': model_output.get('prior_support'),
        'completion_color_gate': model_output.get('completion_color_gate'),
        'completion_intensity_gate': model_output.get('completion_intensity_gate'),
        'completion_scale': model_output.get('completion_scale'),
        'semantic_agreement': model_output.get('semantic_agreement'),
        'depth_agreement': model_output.get('depth_agreement'),
        'normal_agreement': model_output.get('normal_agreement'),
        'geometry_agreement': model_output.get('geometry_agreement'),
        'semantic_quality_prior': model_output.get('semantic_quality_prior'),
        'depth_quality_prior': model_output.get('depth_quality_prior'),
        'normal_quality_prior': model_output.get('normal_quality_prior'),
    }

    for folder, tensor in aux_maps.items():
        if tensor is None:
            continue
        if crop_hw is not None:
            h, w = crop_hw
            tensor = tensor[:, :, :h, :w]
        target_dir = os.path.join(aux_root, folder)
        os.makedirs(target_dir, exist_ok=True)
        save_single_map(tensor, os.path.join(target_dir, file_name))

    route_weights = model_output.get('route_weights', [])
    route_strengths = model_output.get('route_strengths', [])
    target_size = model_output['rgb'].shape[-2:]
    if crop_hw is not None:
        target_size = crop_hw

    merged_route = average_route_stack(route_weights, target_size)
    if merged_route is not None:
        route_names = ['route_hvi', 'route_sem', 'route_geo']
        for idx, route_name in enumerate(route_names):
            target_dir = os.path.join(aux_root, route_name)
            os.makedirs(target_dir, exist_ok=True)
            save_single_map(merged_route[:, idx:idx + 1], os.path.join(target_dir, file_name))

    merged_strength = average_route_stack(route_strengths, target_size)
    if merged_strength is not None:
        target_dir = os.path.join(aux_root, 'route_strength')
        os.makedirs(target_dir, exist_ok=True)
        save_single_map(merged_strength, os.path.join(target_dir, file_name))


def resolve_tta_transforms(tta_mode='none', tta_flip=False):
    if tta_mode is None or tta_mode == 'none':
        tta_mode = 'hflip' if tta_flip else 'none'
    if tta_mode == 'none':
        return [()]
    if tta_mode == 'hflip':
        return [(), (3,)]
    if tta_mode == 'flip4':
        return [(), (3,), (2,), (2, 3)]
    raise ValueError(f'Unsupported tta_mode: {tta_mode}')


def autocast_context(device, enabled=False):
    if enabled and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def crop_priors(priors, top, bottom, left, right):
    if not isinstance(priors, dict):
        return priors
    cropped = {}
    for key, value in priors.items():
        if torch.is_tensor(value) and value.dim() >= 4:
            cropped[key] = value[:, :, top:bottom, left:right]
        else:
            cropped[key] = value
    return cropped


def build_tile_weight(height, width, overlap, top, bottom, left, right, full_h, full_w, device, dtype):
    overlap = max(int(overlap), 0)
    if overlap <= 0:
        return torch.ones((1, 1, height, width), device=device, dtype=dtype)

    y_weight = torch.ones(height, device=device, dtype=dtype)
    x_weight = torch.ones(width, device=device, dtype=dtype)

    y_ramp = min(overlap, height // 2)
    x_ramp = min(overlap, width // 2)

    if top > 0 and y_ramp > 0:
        y_weight[:y_ramp] = torch.linspace(1e-3, 1.0, steps=y_ramp, device=device, dtype=dtype)
    if bottom < full_h and y_ramp > 0:
        y_weight[-y_ramp:] = torch.minimum(
            y_weight[-y_ramp:],
            torch.linspace(1.0, 1e-3, steps=y_ramp, device=device, dtype=dtype),
        )
    if left > 0 and x_ramp > 0:
        x_weight[:x_ramp] = torch.linspace(1e-3, 1.0, steps=x_ramp, device=device, dtype=dtype)
    if right < full_w and x_ramp > 0:
        x_weight[-x_ramp:] = torch.minimum(
            x_weight[-x_ramp:],
            torch.linspace(1.0, 1e-3, steps=x_ramp, device=device, dtype=dtype),
        )

    weight = y_weight.view(1, 1, height, 1) * x_weight.view(1, 1, 1, width)
    return weight.clamp_min(1e-3)


def forward_with_tta(model, input_tensor, priors=None, gamma=1.0, return_aux=False, tta_mode='none', tta_flip=False, amp=False):
    with torch.no_grad():
        transforms = resolve_tta_transforms(tta_mode=tta_mode, tta_flip=tta_flip)
        merged_result = None
        model_device = next(model.parameters()).device
        for dims in transforms:
            if dims:
                aug_input = torch.flip(input_tensor, dims=list(dims))
                aug_priors = flip_priors(priors, dims)
            else:
                aug_input = input_tensor
                aug_priors = priors
            with autocast_context(model_device, enabled=amp):
                aug_result = model(aug_input**gamma, priors=aug_priors, return_aux=return_aux)
            if dims:
                aug_result = _flip_back_tensor(aug_result, dims)
            merged_result = aug_result if merged_result is None else _sum_outputs(merged_result, aug_result)
        return _divide_output(merged_result, max(len(transforms), 1))


def tiled_forward(model, input_tensor, priors=None, gamma=1.0, tta_mode='none', tta_flip=False, tile_size=0, tile_overlap=32, amp=False):
    if tile_size is None or int(tile_size) <= 0:
        return forward_with_tta(model, input_tensor, priors=priors, gamma=gamma, return_aux=False, tta_mode=tta_mode, tta_flip=tta_flip, amp=amp)

    _, _, full_h, full_w = input_tensor.shape
    tile_size = int(tile_size)
    if full_h <= tile_size and full_w <= tile_size:
        return forward_with_tta(model, input_tensor, priors=priors, gamma=gamma, return_aux=False, tta_mode=tta_mode, tta_flip=tta_flip, amp=amp)

    stride = max(tile_size - int(tile_overlap), 1)
    model_device = next(model.parameters()).device
    output_acc = None
    weight_acc = None

    y_positions = list(range(0, max(full_h - tile_size, 0) + 1, stride))
    x_positions = list(range(0, max(full_w - tile_size, 0) + 1, stride))
    if not y_positions or y_positions[-1] != max(full_h - tile_size, 0):
        y_positions.append(max(full_h - tile_size, 0))
    if not x_positions or x_positions[-1] != max(full_w - tile_size, 0):
        x_positions.append(max(full_w - tile_size, 0))

    for top in y_positions:
        for left in x_positions:
            bottom = min(top + tile_size, full_h)
            right = min(left + tile_size, full_w)

            input_tile = input_tensor[:, :, top:bottom, left:right]
            priors_tile = crop_priors(priors, top, bottom, left, right)
            tile_output = forward_with_tta(
                model,
                input_tile,
                priors=priors_tile,
                gamma=gamma,
                return_aux=False,
                tta_mode=tta_mode,
                tta_flip=tta_flip,
                amp=amp,
            )
            if isinstance(tile_output, dict):
                tile_output = tile_output['rgb']

            if output_acc is None:
                channels = tile_output.shape[1]
                output_acc = torch.zeros((1, channels, full_h, full_w), device=model_device, dtype=tile_output.dtype)
                weight_acc = torch.zeros((1, 1, full_h, full_w), device=model_device, dtype=tile_output.dtype)

            weight = build_tile_weight(
                bottom - top,
                right - left,
                tile_overlap,
                top,
                bottom,
                left,
                right,
                full_h,
                full_w,
                model_device,
                tile_output.dtype,
            )
            output_acc[:, :, top:bottom, left:right] += tile_output * weight
            weight_acc[:, :, top:bottom, left:right] += weight

    return output_acc / weight_acc.clamp_min(1e-6)


def eval(model, testing_data_loader, model_path, output_folder,norm_size=True,LOL=False,v2=False,unpaired=False,alpha=1.0,gamma=1.0,dump_aux=False,tta_flip=False,tta_mode='none', amp=False, tile_size=0, tile_overlap=32):
    torch.set_grad_enabled(False)
    if model_path is not None:
        load_checkpoint_compatible(model, model_path)
    model.eval()
    print('Evaluation:')
    model_device = next(model.parameters()).device
    prev_cudnn_enabled = torch.backends.cudnn.enabled
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    try:
        for batch in tqdm(testing_data_loader):
            with torch.no_grad():
                priors = None
                if norm_size:
                    input, name = batch[0], batch[1]
                    if len(batch) >= 3 and isinstance(batch[2], dict):
                        priors = batch[2]
                else:
                    input, name, h, w = batch[0], batch[1], batch[2], batch[3]
                    if len(batch) >= 5 and isinstance(batch[4], dict):
                        priors = batch[4]

                input = input.to(model_device).contiguous()
                if isinstance(priors, dict):
                    priors = {
                        key: value.to(model_device) if torch.is_tensor(value) else value
                        for key, value in priors.items()
                    }
                    if tile_size and int(tile_size) > 0:
                        if dump_aux:
                            raise ValueError('Tile inference does not support --dump_aux_maps yet.')
                        model_result = tiled_forward(model, input, priors=priors, gamma=gamma, tta_mode=tta_mode, tta_flip=tta_flip, tile_size=tile_size, tile_overlap=tile_overlap, amp=amp)
                    else:
                        model_result = forward_with_tta(model, input, priors=priors, gamma=gamma, return_aux=dump_aux, tta_mode=tta_mode, tta_flip=tta_flip, amp=amp)
                else:
                    if tile_size and int(tile_size) > 0:
                        if dump_aux:
                            raise ValueError('Tile inference does not support --dump_aux_maps yet.')
                        model_result = tiled_forward(model, input, priors=None, gamma=gamma, tta_mode=tta_mode, tta_flip=tta_flip, tile_size=tile_size, tile_overlap=tile_overlap, amp=amp)
                    else:
                        model_result = forward_with_tta(model, input, priors=None, gamma=gamma, return_aux=dump_aux, tta_mode=tta_mode, tta_flip=tta_flip, amp=amp)

                if isinstance(model_result, dict):
                    output = model_result['rgb']
                else:
                    output = model_result

            os.makedirs(output_folder, exist_ok=True)

            output = torch.clamp(output, 0, 1)
            crop_hw = None
            if not norm_size:
                if torch.is_tensor(h):
                    h = int(h.flatten()[0].item())
                else:
                    h = int(h)
                if torch.is_tensor(w):
                    w = int(w.flatten()[0].item())
                else:
                    w = int(w)
                output = output[:, :, :h, :w]
                crop_hw = (h, w)

            output_img = transforms.ToPILImage()(output.squeeze(0))
            output_img.save(os.path.join(output_folder, name[0]))
            if dump_aux and isinstance(model_result, dict):
                dump_auxiliary_outputs(output_folder, name[0], model_result, crop_hw=crop_hw)
            torch.cuda.empty_cache()
        print('===> End evaluation')
    finally:
        if LOL:
            model.trans.gated = False
        elif v2:
            model.trans.gated2 = False
        torch.backends.cudnn.enabled = prev_cudnn_enabled
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark
        torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    eval_parser = argparse.ArgumentParser(description='Eval')
    eval_parser.add_argument('--cpu', action='store_true', help='CPU-only inference')
    eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
    eval_parser.add_argument('--weights_path', type=str, default=None, help='custom checkpoint path, e.g. ./weights/train/epoch_10.pth')
    eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
    eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
    eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
    eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
    eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')
    eval_parser.add_argument('--fivek', action='store_true', help='output FiveK dataset')

    eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
    eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
    eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

    eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
    eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
    eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
    eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
    eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
    eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
    eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
    eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
    eval_parser.add_argument('--alpha', type=float, default=1.0)
    eval_parser.add_argument('--gamma', type=float, default=1.0)
    eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth')
    eval_parser.add_argument('--semantic_prior_dir', type=str, default='')
    eval_parser.add_argument('--depth_prior_dir', type=str, default='')
    eval_parser.add_argument('--normal_prior_dir', type=str, default='')
    eval_parser.add_argument('--semantic_quality_dir', type=str, default='')
    eval_parser.add_argument('--depth_quality_dir', type=str, default='')
    eval_parser.add_argument('--normal_quality_dir', type=str, default='')
    eval_parser.add_argument('--strict_prior_loading', action='store_true')
    eval_parser.add_argument('--semantic_in_channels', type=int, default=3)
    eval_parser.add_argument('--geometry_in_channels', type=int, default=4)
    eval_parser.add_argument('--lite_mode', type=str, default='none', choices=['none', 'safe'])
    eval_parser.add_argument('--dump_aux_maps', action='store_true', help='save observability/reliability/route maps')
    eval_parser.add_argument('--tta_flip', action='store_true', help='use horizontal flip self-ensemble during evaluation')
    eval_parser.add_argument('--tta_mode', type=str, default='none', choices=['none', 'hflip', 'flip4'], help='self-ensemble mode during evaluation')
    eval_parser.add_argument('--num_workers', type=int, default=1, help='number of dataloader workers during evaluation')
    eval_parser.add_argument('--amp', action='store_true', help='use mixed precision on CUDA during evaluation')
    eval_parser.add_argument('--tile_size', type=int, default=0, help='tile size for sliding-window inference; 0 disables tiling')
    eval_parser.add_argument('--tile_overlap', type=int, default=32, help='tile overlap for sliding-window inference')

    ep = eval_parser.parse_args()


    use_cuda = not ep.cpu
    if use_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = ep.num_workers
    alpha = None
    prior_config = build_prior_config(
        semantic_dir=ep.semantic_prior_dir,
        depth_dir=ep.depth_prior_dir,
        normal_dir=ep.normal_prior_dir,
        semantic_quality_dir=ep.semantic_quality_dir,
        depth_quality_dir=ep.depth_quality_dir,
        normal_quality_dir=ep.normal_quality_dir,
        strict=ep.strict_prior_loading,
    )
    if any([
        ep.semantic_prior_dir,
        ep.depth_prior_dir,
        ep.normal_prior_dir,
        ep.semantic_quality_dir,
        ep.depth_quality_dir,
        ep.normal_quality_dir,
    ]) and not prior_config.enabled:
        raise ValueError('Failed to index the provided prior directories.')
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLdataset/eval15/low", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        if ep.perc:
            weight_path = './weights/LOLv1/w_perc.pth'
        else:
            weight_path = './weights/LOLv1/wo_perc.pth'
        
            
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        if ep.best_GT_mean:
            weight_path = './weights/LOLv2_real/w_perc.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = './weights/LOLv2_real/best_PSNR.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = './weights/LOLv2_real/best_SSIM.pth'
            alpha = 0.82
        else:
            alpha = ep.alpha
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        if ep.perc:
            weight_path = './weights/LOLv2_syn/w_perc.pth'
        else:
            weight_path = './weights/LOLv2_syn/wo_perc.pth'
            
    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_grad/'
        weight_path = './weights/SICE.pth'
        norm_size = False
        
    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_mix/'
        weight_path = './weights/SICE.pth'
        norm_size = False
        
    elif ep.fivek:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/FiveK/test/input", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/fivek/'
        weight_path = './weights/fivek.pth'
        norm_size = False
    
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/DICM", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/LIME", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/MEF", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/NPE", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/VV", prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/VV/'
        elif ep.custome:
            eval_data = DataLoader(dataset=get_SICE_eval_set(ep.custome_path, prior_config=prior_config), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/custome/'
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights

    if ep.weights_path is not None:
        weight_path = ep.weights_path
        
    eval_net = DAMG(
        semantic_in_channels=ep.semantic_in_channels,
        geometry_in_channels=ep.geometry_in_channels,
        lite_mode=ep.lite_mode,
    )
    if use_cuda:
        eval_net = eval_net.cuda()
    eval(
        eval_net,
        eval_data,
        weight_path,
        output_folder,
        norm_size=norm_size,
        LOL=ep.lol,
        v2=ep.lol_v2_real,
        unpaired=ep.unpaired,
        alpha=alpha,
        gamma=ep.gamma,
        dump_aux=ep.dump_aux_maps,
        tta_flip=ep.tta_flip,
        tta_mode=ep.tta_mode,
        amp=ep.amp,
        tile_size=ep.tile_size,
        tile_overlap=ep.tile_overlap,
    )
