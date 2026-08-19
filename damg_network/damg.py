import torch
import torch.nn as nn
import torch.nn.functional as F
from damg_network.hvi_photometric import RGB_HVI
from damg_network.semantic_geometric_priors import PriorPyramidEncoder, ConfidenceEstimator, ReconstructionHead, RefinementHead, PriorFusionBlock
from damg_network.feature_transformer import *
from damg_network.adaptive_fusion import *

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    class PyTorchModelHubMixin:
        pass

class DAMG(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 semantic_in_channels=3,
                 geometry_in_channels=4,
                 head_refine_blocks=3,
                 rgb_refine_blocks=4,
                 lite_mode='none',
        ):
        super(DAMG, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        self.lite_mode = str(lite_mode or 'none').lower()
        if self.lite_mode not in ('none', 'safe'):
            raise ValueError(f'Unsupported lite_mode: {lite_mode}')
        estimator_block_type = 'separable' if self.lite_mode == 'safe' else 'standard'
        fusion_block_type = 'separable' if self.lite_mode == 'safe' else 'standard'
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        self.semantic_in_channels = semantic_in_channels
        self.geometry_in_channels = geometry_in_channels
        self.semantic_encoder = PriorPyramidEncoder(semantic_in_channels, channels, norm=norm)
        self.depth_encoder = PriorPyramidEncoder(1, channels, norm=norm)
        self.normal_encoder = PriorPyramidEncoder(3, channels, norm=norm)
        self.geometry_fusion_full = PriorFusionBlock(
            ch1 * 2,
            ch1,
            out_channels=ch1,
            block_type=fusion_block_type,
        )
        self.geometry_fusion_pyramid = nn.ModuleList([
            PriorFusionBlock(ch2 * 2, ch2, out_channels=ch2, block_type=fusion_block_type),
            PriorFusionBlock(ch3 * 2, ch3, out_channels=ch3, block_type=fusion_block_type),
            PriorFusionBlock(ch4 * 2, ch4, out_channels=ch4, block_type=fusion_block_type),
        ])

        obs_in_channels = 3 + 3 + (ch1 * 2)
        quality_in_channels = (ch1 * 3) + 3
        reliability_in_channels = (ch1 * 3) + 4
        self.obs_estimator = ConfidenceEstimator(obs_in_channels, ch1, block_type=estimator_block_type)
        self.semantic_quality_estimator = ConfidenceEstimator(quality_in_channels, ch1, block_type=estimator_block_type)
        self.depth_quality_estimator = ConfidenceEstimator(quality_in_channels, ch1, block_type=estimator_block_type)
        self.normal_quality_estimator = ConfidenceEstimator(quality_in_channels, ch1, block_type=estimator_block_type)
        self.semantic_reliability_estimator = ConfidenceEstimator(reliability_in_channels, ch1, block_type=estimator_block_type)
        self.geometry_reliability_estimator = ConfidenceEstimator(reliability_in_channels, ch1, block_type=estimator_block_type)
        self.obs_pyramid_estimators = nn.ModuleList([
            ConfidenceEstimator((ch2 * 2) + 4, ch2, block_type=estimator_block_type),
            ConfidenceEstimator((ch3 * 2) + 4, ch3, block_type=estimator_block_type),
            ConfidenceEstimator((ch4 * 2) + 4, ch4, block_type=estimator_block_type),
        ])
        self.semantic_quality_pyramid_estimators = nn.ModuleList([
            ConfidenceEstimator((ch2 * 3) + 3, ch2, block_type=estimator_block_type),
            ConfidenceEstimator((ch3 * 3) + 3, ch3, block_type=estimator_block_type),
            ConfidenceEstimator((ch4 * 3) + 3, ch4, block_type=estimator_block_type),
        ])
        self.depth_quality_pyramid_estimators = nn.ModuleList([
            ConfidenceEstimator((ch2 * 3) + 3, ch2, block_type=estimator_block_type),
            ConfidenceEstimator((ch3 * 3) + 3, ch3, block_type=estimator_block_type),
            ConfidenceEstimator((ch4 * 3) + 3, ch4, block_type=estimator_block_type),
        ])
        self.normal_quality_pyramid_estimators = nn.ModuleList([
            ConfidenceEstimator((ch2 * 3) + 3, ch2, block_type=estimator_block_type),
            ConfidenceEstimator((ch3 * 3) + 3, ch3, block_type=estimator_block_type),
            ConfidenceEstimator((ch4 * 3) + 3, ch4, block_type=estimator_block_type),
        ])
        self.semantic_reliability_pyramid_estimators = nn.ModuleList([
            ConfidenceEstimator((ch2 * 3) + 4, ch2, block_type=estimator_block_type),
            ConfidenceEstimator((ch3 * 3) + 4, ch3, block_type=estimator_block_type),
            ConfidenceEstimator((ch4 * 3) + 4, ch4, block_type=estimator_block_type),
        ])
        self.geometry_reliability_pyramid_estimators = nn.ModuleList([
            ConfidenceEstimator((ch2 * 3) + 4, ch2, block_type=estimator_block_type),
            ConfidenceEstimator((ch3 * 3) + 4, ch3, block_type=estimator_block_type),
            ConfidenceEstimator((ch4 * 3) + 4, ch4, block_type=estimator_block_type),
        ])

        self.observable_head = ReconstructionHead((ch1 * 2) + 1, ch1, out_channels=3)
        self.unobservable_head = ReconstructionHead((ch1 * 4) + 6, ch1, out_channels=3)
        self.observable_refiner = RefinementHead(((ch1 * 2) + 1) + 3, ch1, out_channels=3, num_blocks=head_refine_blocks)
        self.unobservable_refiner = RefinementHead(((ch1 * 4) + 6) + 3, ch1, out_channels=3, num_blocks=head_refine_blocks)
        refine_in_channels = 3 + 3 + 6
        self.rgb_refiner = RefinementHead(refine_in_channels, ch1, out_channels=3, num_blocks=rgb_refine_blocks)
        self.exposure_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, ch1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch1, ch1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(ch1, 6, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.prior_support_fusion = ConfidenceEstimator(7, ch1, block_type=estimator_block_type)
        
        self.trans = RGB_HVI()

    def _match_channels(self, x, channels):
        if x.shape[1] == channels:
            return x
        if x.shape[1] > channels:
            return x[:, :channels, :, :]
        repeat = (channels + x.shape[1] - 1) // x.shape[1]
        x = x.repeat(1, repeat, 1, 1)
        return x[:, :channels, :, :]

    def _gradient_map(self, x):
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        return grad_x + grad_y

    def _pseudo_normal(self, depth):
        grad_x = F.pad(depth[:, :, :, 1:] - depth[:, :, :, :-1], (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(depth[:, :, 1:, :] - depth[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')
        normal = torch.cat([-grad_x, -grad_y, torch.ones_like(depth)], dim=1)
        return F.normalize(normal, dim=1)

    def _prepare_priors(self, x, priors=None):
        priors = {} if priors is None else priors
        semantic = priors.get('semantic', x)
        geometry = priors.get('geometry')
        depth = priors.get('depth')
        normal = priors.get('normal')
        semantic_quality = priors.get('semantic_quality')
        depth_quality = priors.get('depth_quality')
        normal_quality = priors.get('normal_quality')
        semantic_valid = priors.get('semantic_valid')
        depth_valid = priors.get('depth_valid')
        normal_valid = priors.get('normal_valid')

        if geometry is not None and torch.is_tensor(geometry):
            geometry = geometry.to(device=x.device, dtype=x.dtype)
            if depth is None and geometry.shape[1] >= 1:
                depth = geometry[:, 0:1, :, :]
            if normal is None and geometry.shape[1] >= 4:
                normal = geometry[:, 1:4, :, :]

        if depth is None:
            depth = x.mean(dim=1, keepdim=True)
        if normal is None:
            normal = self._pseudo_normal(depth)
        if geometry is None:
            geometry = torch.cat([depth, normal], dim=1)

        if semantic_valid is None:
            semantic_valid = torch.ones_like(depth)
        if depth_valid is None:
            depth_valid = torch.ones_like(depth)
        if normal_valid is None:
            normal_valid = torch.ones_like(depth)
        geometry_valid = torch.clamp(torch.maximum(depth_valid, normal_valid), 0.0, 1.0)

        semantic = self._match_channels(semantic.to(device=x.device, dtype=x.dtype), self.semantic_in_channels)
        geometry = self._match_channels(geometry.to(device=x.device, dtype=x.dtype), self.geometry_in_channels)
        depth = depth.to(device=x.device, dtype=x.dtype)
        normal = normal.to(device=x.device, dtype=x.dtype)
        if semantic_quality is not None:
            semantic_quality = semantic_quality.to(device=x.device, dtype=x.dtype)
        if depth_quality is not None:
            depth_quality = depth_quality.to(device=x.device, dtype=x.dtype)
        if normal_quality is not None:
            normal_quality = normal_quality.to(device=x.device, dtype=x.dtype)
        semantic_valid = semantic_valid.to(device=x.device, dtype=x.dtype)
        depth_valid = depth_valid.to(device=x.device, dtype=x.dtype)
        normal_valid = normal_valid.to(device=x.device, dtype=x.dtype)
        geometry_valid = geometry_valid.to(device=x.device, dtype=x.dtype)
        return (
            semantic,
            geometry,
            depth,
            normal,
            semantic_quality,
            depth_quality,
            normal_quality,
            semantic_valid,
            depth_valid,
            normal_valid,
            geometry_valid,
        )

    def _resize_confidence(self, conf_map, target):
        return F.interpolate(conf_map, size=target.shape[-2:], mode='bilinear', align_corners=False)

    def _resize_to_size(self, tensor, size):
        return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

    def _blend_multiscale(self, tensors, size):
        if not tensors:
            raise ValueError('Expected at least one tensor to blend.')
        blended = None
        for tensor in tensors:
            resized = tensor if tensor.shape[-2:] == size else self._resize_to_size(tensor, size)
            blended = resized if blended is None else blended + resized
        return torch.clamp(blended / float(len(tensors)), 0.0, 1.0)

    def _feature_agreement(self, feat_a, feat_b):
        if feat_a.shape[-2:] != feat_b.shape[-2:]:
            feat_b = F.interpolate(feat_b, size=feat_a.shape[-2:], mode='bilinear', align_corners=False)
        feat_a = F.normalize(feat_a, dim=1)
        feat_b = F.normalize(feat_b, dim=1)
        return torch.clamp((feat_a * feat_b).sum(dim=1, keepdim=True), -1.0, 1.0).add(1.0).mul(0.5)

    def _estimate_obs_level(self, estimator, hv_feat, i_feat, stat_maps, prev_obs):
        resized_stats = [self._resize_confidence(stat_map, hv_feat) for stat_map in stat_maps]
        obs_base = estimator(torch.cat([hv_feat, i_feat] + resized_stats, dim=1))
        prev_obs = self._resize_confidence(prev_obs, hv_feat)
        return torch.clamp((0.65 * obs_base) + (0.35 * prev_obs), 0.0, 1.0)

    def _estimate_prior_quality(self, estimator, prev_map, feat_a, feat_b, prior_feat, obs_map, valid_map, prior_quality=None):
        agreement = 0.5 * (self._feature_agreement(feat_a, prior_feat) + self._feature_agreement(feat_b, prior_feat))
        if valid_map.shape[-2:] != agreement.shape[-2:]:
            valid_map = self._resize_confidence(valid_map, feat_a)
        obs_map = self._resize_confidence(obs_map, feat_a)
        if prior_quality is not None:
            prior_quality = self._resize_confidence(prior_quality, feat_a)
        quality_base = estimator(torch.cat([feat_a, feat_b, prior_feat, obs_map, agreement, valid_map], dim=1))
        if prev_map is None:
            if prior_quality is None:
                refined = torch.clamp(
                    (0.45 * quality_base) +
                    (0.25 * agreement) +
                    (0.20 * valid_map) +
                    (0.10 * obs_map),
                    0.0,
                    1.0,
                )
            else:
                refined = torch.clamp(
                    (0.35 * quality_base) +
                    (0.20 * agreement) +
                    (0.15 * valid_map) +
                    (0.10 * obs_map) +
                    (0.20 * prior_quality),
                    0.0,
                    1.0,
                )
            return refined, agreement
        prev_map = self._resize_confidence(prev_map, feat_a)
        if prior_quality is None:
            refined = torch.clamp(
                (0.40 * quality_base) +
                (0.20 * agreement) +
                (0.15 * valid_map) +
                (0.15 * obs_map) +
                (0.10 * prev_map),
                0.0,
                1.0,
            )
        else:
            refined = torch.clamp(
                (0.32 * quality_base) +
                (0.18 * agreement) +
                (0.12 * valid_map) +
                (0.12 * obs_map) +
                (0.12 * prev_map) +
                (0.14 * prior_quality),
                0.0,
                1.0,
            )
        return refined, agreement

    def _estimate_reliability_level(self, estimator, prev_map, feat_a, feat_b, prior_feat, obs_map, valid_map, quality_map):
        agreement = 0.5 * (self._feature_agreement(feat_a, prior_feat) + self._feature_agreement(feat_b, prior_feat))
        if valid_map.shape[-2:] != agreement.shape[-2:]:
            valid_map = self._resize_confidence(valid_map, feat_a)
        if quality_map.shape[-2:] != agreement.shape[-2:]:
            quality_map = self._resize_confidence(quality_map, feat_a)
        obs_map = self._resize_confidence(obs_map, feat_a)
        reliability_base = estimator(torch.cat([feat_a, feat_b, prior_feat, obs_map, agreement, valid_map, quality_map], dim=1))
        prev_map = self._resize_confidence(prev_map, feat_a)
        refined = torch.clamp(
            (0.40 * reliability_base) +
            (0.20 * agreement) +
            (0.15 * valid_map) +
            (0.15 * quality_map) +
            (0.10 * prev_map),
            0.0,
            1.0,
        )
        return refined, agreement

    def _combine_prior_maps(self, map_a, map_b, valid_a, valid_b):
        valid_a = self._resize_confidence(valid_a, map_a)
        valid_b = self._resize_confidence(valid_b, map_b)
        weighted_a = map_a * valid_a
        weighted_b = map_b * valid_b
        denom = weighted_a + weighted_b + 1e-6
        weight_a = weighted_a / denom
        weight_b = weighted_b / denom
        combined = torch.clamp((weight_a * map_a) + (weight_b * map_b), 0.0, 1.0)
        return combined, weight_a, weight_b

    def _fuse_geometry_features(self, fusion_block, depth_feat, normal_feat, depth_quality, normal_quality):
        depth_quality = self._resize_confidence(depth_quality, depth_feat)
        normal_quality = self._resize_confidence(normal_quality, normal_feat)
        denom = depth_quality + normal_quality + 1e-6
        depth_weight = depth_quality / denom
        normal_weight = normal_quality / denom
        return fusion_block(torch.cat([depth_feat * depth_weight, normal_feat * normal_weight], dim=1))
        
    def forward(self, x, priors=None, return_aux=False, completion_scale=1.0):
        dtypes = x.dtype
        completion_scale = float(max(0.0, min(1.0, completion_scale)))
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        (
            semantic_prior,
            geometry_prior,
            depth_prior,
            normal_prior,
            semantic_quality_prior,
            depth_quality_prior,
            normal_quality_prior,
            semantic_valid,
            depth_valid,
            normal_valid,
            geometry_valid,
        ) = self._prepare_priors(x, priors)

        semantic_feats = self.semantic_encoder(semantic_prior)
        depth_feats = self.depth_encoder(depth_prior)
        normal_feats = self.normal_encoder(normal_prior)

        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        hv_magnitude = torch.sqrt(hvi[:, 0:1, :, :] ** 2 + hvi[:, 1:2, :, :] ** 2 + 1e-6)
        local_contrast = torch.abs(i - F.avg_pool2d(i, kernel_size=5, stride=1, padding=2))
        local_gradient = self._gradient_map(i)
        obs_stats = [i, hv_magnitude, local_contrast, local_gradient]
        obs_input = torch.cat([hvi, hv_magnitude, local_contrast, local_gradient, hv_0, i_enc0], dim=1)
        obs_full = self.obs_estimator(obs_input)
        semantic_valid_full = self._resize_confidence(semantic_valid, hv_0)
        depth_valid_full = self._resize_confidence(depth_valid, hv_0)
        normal_valid_full = self._resize_confidence(normal_valid, hv_0)
        geometry_valid_full = self._resize_confidence(geometry_valid, hv_0)
        semantic_agreement_full = 0.5 * (
            self._feature_agreement(hv_0, semantic_feats['full']) +
            self._feature_agreement(i_enc0, semantic_feats['full'])
        )
        semantic_quality_full, _ = self._estimate_prior_quality(
            self.semantic_quality_estimator,
            None,
            hv_0,
            i_enc0,
            semantic_feats['full'],
            obs_full,
            semantic_valid_full,
            prior_quality=semantic_quality_prior,
        )
        depth_quality_full, depth_agreement_full = self._estimate_prior_quality(
            self.depth_quality_estimator,
            None,
            hv_0,
            i_enc0,
            depth_feats['full'],
            obs_full,
            depth_valid_full,
            prior_quality=depth_quality_prior,
        )
        normal_quality_full, normal_agreement_full = self._estimate_prior_quality(
            self.normal_quality_estimator,
            None,
            hv_0,
            i_enc0,
            normal_feats['full'],
            obs_full,
            normal_valid_full,
            prior_quality=normal_quality_prior,
        )
        geometry_quality_full, depth_mix_full, normal_mix_full = self._combine_prior_maps(
            depth_quality_full,
            normal_quality_full,
            depth_valid_full,
            normal_valid_full,
        )
        geometry_agreement_full = torch.clamp(
            (depth_mix_full * depth_agreement_full) + (normal_mix_full * normal_agreement_full),
            0.0,
            1.0,
        )
        geometry_feats_full = self._fuse_geometry_features(
            self.geometry_fusion_full,
            depth_feats['full'],
            normal_feats['full'],
            depth_quality_full,
            normal_quality_full,
        )
        rs_full = self.semantic_reliability_estimator(
            torch.cat([
                hv_0,
                i_enc0,
                semantic_feats['full'],
                obs_full,
                semantic_agreement_full,
                semantic_valid_full,
                semantic_quality_full,
            ], dim=1)
        )
        rg_full = self.geometry_reliability_estimator(
            torch.cat([
                hv_0,
                i_enc0,
                geometry_feats_full,
                obs_full,
                geometry_agreement_full,
                geometry_valid_full,
                geometry_quality_full,
            ], dim=1)
        )
        rs_full = torch.clamp(
            (0.55 * rs_full) + (0.20 * semantic_agreement_full) + (0.20 * semantic_quality_full) + (0.05 * semantic_valid_full),
            0.0,
            1.0,
        )
        rg_full = torch.clamp(
            (0.50 * rg_full) + (0.20 * geometry_agreement_full) + (0.20 * geometry_quality_full) + (0.10 * geometry_valid_full),
            0.0,
            1.0,
        )
        obs_level1 = self._estimate_obs_level(self.obs_pyramid_estimators[0], hv_1, i_enc1, obs_stats, obs_full)

        semantic_valid_level1 = self._resize_confidence(semantic_valid, hv_1)
        depth_valid_level1 = self._resize_confidence(depth_valid, hv_1)
        normal_valid_level1 = self._resize_confidence(normal_valid, hv_1)
        geometry_valid_level1 = self._resize_confidence(geometry_valid, hv_1)
        semantic_quality_level1, semantic_agreement_level1 = self._estimate_prior_quality(
            self.semantic_quality_pyramid_estimators[0],
            semantic_quality_full,
            hv_1,
            i_enc1,
            semantic_feats['pyramid'][0],
            obs_level1,
            semantic_valid_level1,
        )
        depth_quality_level1, depth_agreement_level1 = self._estimate_prior_quality(
            self.depth_quality_pyramid_estimators[0],
            depth_quality_full,
            hv_1,
            i_enc1,
            depth_feats['pyramid'][0],
            obs_level1,
            depth_valid_level1,
        )
        normal_quality_level1, normal_agreement_level1 = self._estimate_prior_quality(
            self.normal_quality_pyramid_estimators[0],
            normal_quality_full,
            hv_1,
            i_enc1,
            normal_feats['pyramid'][0],
            obs_level1,
            normal_valid_level1,
        )
        geometry_quality_level1, depth_mix_level1, normal_mix_level1 = self._combine_prior_maps(
            depth_quality_level1,
            normal_quality_level1,
            depth_valid_level1,
            normal_valid_level1,
        )
        geometry_agreement_level1 = torch.clamp(
            (depth_mix_level1 * depth_agreement_level1) + (normal_mix_level1 * normal_agreement_level1),
            0.0,
            1.0,
        )
        geometry_feats_level1 = self._fuse_geometry_features(
            self.geometry_fusion_pyramid[0],
            depth_feats['pyramid'][0],
            normal_feats['pyramid'][0],
            depth_quality_level1,
            normal_quality_level1,
        )
        rs_level1, semantic_agreement_level1 = self._estimate_reliability_level(
            self.semantic_reliability_pyramid_estimators[0],
            rs_full,
            hv_1,
            i_enc1,
            semantic_feats['pyramid'][0],
            obs_level1,
            semantic_valid_level1,
            semantic_quality_level1,
        )
        rg_level1, geometry_agreement_level1 = self._estimate_reliability_level(
            self.geometry_reliability_pyramid_estimators[0],
            rg_full,
            hv_1,
            i_enc1,
            geometry_feats_level1,
            obs_level1,
            geometry_valid_level1,
            geometry_quality_level1,
        )
        
        i_enc2, route_i1, strength_i1 = self.I_LCA1(
            i_enc1, hv_1, semantic_feats['pyramid'][0], geometry_feats_level1,
            obs_level1, rs_level1, rg_level1,
            semantic_valid_level1 * semantic_quality_level1,
            geometry_valid_level1 * geometry_quality_level1,
            return_route=True
        )
        hv_2, route_h1, strength_h1 = self.HV_LCA1(
            hv_1, i_enc1, semantic_feats['pyramid'][0], geometry_feats_level1,
            obs_level1, rs_level1, rg_level1,
            semantic_valid_level1 * semantic_quality_level1,
            geometry_valid_level1 * geometry_quality_level1,
            return_route=True
        )
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        obs_level2 = self._estimate_obs_level(self.obs_pyramid_estimators[1], hv_2, i_enc2, obs_stats, obs_level1)
        semantic_valid_level2 = self._resize_confidence(semantic_valid, hv_2)
        depth_valid_level2 = self._resize_confidence(depth_valid, hv_2)
        normal_valid_level2 = self._resize_confidence(normal_valid, hv_2)
        geometry_valid_level2 = self._resize_confidence(geometry_valid, hv_2)
        semantic_quality_level2, semantic_agreement_level2 = self._estimate_prior_quality(
            self.semantic_quality_pyramid_estimators[1],
            semantic_quality_level1,
            hv_2,
            i_enc2,
            semantic_feats['pyramid'][1],
            obs_level2,
            semantic_valid_level2,
        )
        depth_quality_level2, depth_agreement_level2 = self._estimate_prior_quality(
            self.depth_quality_pyramid_estimators[1],
            depth_quality_level1,
            hv_2,
            i_enc2,
            depth_feats['pyramid'][1],
            obs_level2,
            depth_valid_level2,
        )
        normal_quality_level2, normal_agreement_level2 = self._estimate_prior_quality(
            self.normal_quality_pyramid_estimators[1],
            normal_quality_level1,
            hv_2,
            i_enc2,
            normal_feats['pyramid'][1],
            obs_level2,
            normal_valid_level2,
        )
        geometry_quality_level2, depth_mix_level2, normal_mix_level2 = self._combine_prior_maps(
            depth_quality_level2,
            normal_quality_level2,
            depth_valid_level2,
            normal_valid_level2,
        )
        geometry_agreement_level2 = torch.clamp(
            (depth_mix_level2 * depth_agreement_level2) + (normal_mix_level2 * normal_agreement_level2),
            0.0,
            1.0,
        )
        geometry_feats_level2 = self._fuse_geometry_features(
            self.geometry_fusion_pyramid[1],
            depth_feats['pyramid'][1],
            normal_feats['pyramid'][1],
            depth_quality_level2,
            normal_quality_level2,
        )
        rs_level2, semantic_agreement_level2 = self._estimate_reliability_level(
            self.semantic_reliability_pyramid_estimators[1],
            rs_level1,
            hv_2,
            i_enc2,
            semantic_feats['pyramid'][1],
            obs_level2,
            semantic_valid_level2,
            semantic_quality_level2,
        )
        rg_level2, geometry_agreement_level2 = self._estimate_reliability_level(
            self.geometry_reliability_pyramid_estimators[1],
            rg_level1,
            hv_2,
            i_enc2,
            geometry_feats_level2,
            obs_level2,
            geometry_valid_level2,
            geometry_quality_level2,
        )
        
        i_enc3, route_i2, strength_i2 = self.I_LCA2(
            i_enc2, hv_2, semantic_feats['pyramid'][1], geometry_feats_level2,
            obs_level2, rs_level2, rg_level2,
            semantic_valid_level2 * semantic_quality_level2,
            geometry_valid_level2 * geometry_quality_level2,
            return_route=True
        )
        hv_3, route_h2, strength_h2 = self.HV_LCA2(
            hv_2, i_enc2, semantic_feats['pyramid'][1], geometry_feats_level2,
            obs_level2, rs_level2, rg_level2,
            semantic_valid_level2 * semantic_quality_level2,
            geometry_valid_level2 * geometry_quality_level2,
            return_route=True
        )
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc3)
        hv_3 = self.HVE_block3(hv_3)

        obs_level3 = self._estimate_obs_level(self.obs_pyramid_estimators[2], hv_3, i_enc3, obs_stats, obs_level2)
        semantic_valid_level3 = self._resize_confidence(semantic_valid, hv_3)
        depth_valid_level3 = self._resize_confidence(depth_valid, hv_3)
        normal_valid_level3 = self._resize_confidence(normal_valid, hv_3)
        geometry_valid_level3 = self._resize_confidence(geometry_valid, hv_3)
        semantic_quality_level3, semantic_agreement_level3 = self._estimate_prior_quality(
            self.semantic_quality_pyramid_estimators[2],
            semantic_quality_level2,
            hv_3,
            i_enc3,
            semantic_feats['pyramid'][2],
            obs_level3,
            semantic_valid_level3,
        )
        depth_quality_level3, depth_agreement_level3 = self._estimate_prior_quality(
            self.depth_quality_pyramid_estimators[2],
            depth_quality_level2,
            hv_3,
            i_enc3,
            depth_feats['pyramid'][2],
            obs_level3,
            depth_valid_level3,
        )
        normal_quality_level3, normal_agreement_level3 = self._estimate_prior_quality(
            self.normal_quality_pyramid_estimators[2],
            normal_quality_level2,
            hv_3,
            i_enc3,
            normal_feats['pyramid'][2],
            obs_level3,
            normal_valid_level3,
        )
        geometry_quality_level3, depth_mix_level3, normal_mix_level3 = self._combine_prior_maps(
            depth_quality_level3,
            normal_quality_level3,
            depth_valid_level3,
            normal_valid_level3,
        )
        geometry_agreement_level3 = torch.clamp(
            (depth_mix_level3 * depth_agreement_level3) + (normal_mix_level3 * normal_agreement_level3),
            0.0,
            1.0,
        )
        geometry_feats_level3 = self._fuse_geometry_features(
            self.geometry_fusion_pyramid[2],
            depth_feats['pyramid'][2],
            normal_feats['pyramid'][2],
            depth_quality_level3,
            normal_quality_level3,
        )
        rs_level3, semantic_agreement_level3 = self._estimate_reliability_level(
            self.semantic_reliability_pyramid_estimators[2],
            rs_level2,
            hv_3,
            i_enc3,
            semantic_feats['pyramid'][2],
            obs_level3,
            semantic_valid_level3,
            semantic_quality_level3,
        )
        rg_level3, geometry_agreement_level3 = self._estimate_reliability_level(
            self.geometry_reliability_pyramid_estimators[2],
            rg_level2,
            hv_3,
            i_enc3,
            geometry_feats_level3,
            obs_level3,
            geometry_valid_level3,
            geometry_quality_level3,
        )
        
        i_enc4, route_i3, strength_i3 = self.I_LCA3(
            i_enc3, hv_3, semantic_feats['pyramid'][2], geometry_feats_level3,
            obs_level3, rs_level3, rg_level3,
            semantic_valid_level3 * semantic_quality_level3,
            geometry_valid_level3 * geometry_quality_level3,
            return_route=True
        )
        hv_4, route_h3, strength_h3 = self.HV_LCA3(
            hv_3, i_enc3, semantic_feats['pyramid'][2], geometry_feats_level3,
            obs_level3, rs_level3, rg_level3,
            semantic_valid_level3 * semantic_quality_level3,
            geometry_valid_level3 * geometry_quality_level3,
            return_route=True
        )
        
        i_dec4, route_i4, strength_i4 = self.I_LCA4(
            i_enc4, hv_4, semantic_feats['pyramid'][2], geometry_feats_level3,
            obs_level3, rs_level3, rg_level3,
            semantic_valid_level3 * semantic_quality_level3,
            geometry_valid_level3 * geometry_quality_level3,
            return_route=True
        )
        hv_4, route_h4, strength_h4 = self.HV_LCA4(
            hv_4, i_enc4, semantic_feats['pyramid'][2], geometry_feats_level3,
            obs_level3, rs_level3, rg_level3,
            semantic_valid_level3 * semantic_quality_level3,
            geometry_valid_level3 * geometry_quality_level3,
            return_route=True
        )
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2, route_i5, strength_i5 = self.I_LCA5(
            i_dec3, hv_3, semantic_feats['pyramid'][1], geometry_feats_level2,
            obs_level2, rs_level2, rg_level2,
            semantic_valid_level2 * semantic_quality_level2,
            geometry_valid_level2 * geometry_quality_level2,
            return_route=True
        )
        hv_2, route_h5, strength_h5 = self.HV_LCA5(
            hv_3, i_dec3, semantic_feats['pyramid'][1], geometry_feats_level2,
            obs_level2, rs_level2, rg_level2,
            semantic_valid_level2 * semantic_quality_level2,
            geometry_valid_level2 * geometry_quality_level2,
            return_route=True
        )
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec2, v_jump1)
        
        i_dec1, route_i6, strength_i6 = self.I_LCA6(
            i_dec2, hv_2, semantic_feats['pyramid'][0], geometry_feats_level1,
            obs_level1, rs_level1, rg_level1,
            semantic_valid_level1 * semantic_quality_level1,
            geometry_valid_level1 * geometry_quality_level1,
            return_route=True
        )
        hv_1, route_h6, strength_h6 = self.HV_LCA6(
            hv_2, i_dec2, semantic_feats['pyramid'][0], geometry_feats_level1,
            obs_level1, rs_level1, rg_level1,
            semantic_valid_level1 * semantic_quality_level1,
            geometry_valid_level1 * geometry_quality_level1,
            return_route=True
        )
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        obs_final = self._blend_multiscale([obs_full, obs_level1, obs_level2, obs_level3], obs_full.shape[-2:])
        rs_final = self._blend_multiscale([rs_full, rs_level1, rs_level2, rs_level3], rs_full.shape[-2:])
        rg_final = self._blend_multiscale([rg_full, rg_level1, rg_level2, rg_level3], rg_full.shape[-2:])
        semantic_quality_final = self._blend_multiscale(
            [semantic_quality_full, semantic_quality_level1, semantic_quality_level2, semantic_quality_level3],
            semantic_quality_full.shape[-2:],
        )
        depth_quality_final = self._blend_multiscale(
            [depth_quality_full, depth_quality_level1, depth_quality_level2, depth_quality_level3],
            depth_quality_full.shape[-2:],
        )
        normal_quality_final = self._blend_multiscale(
            [normal_quality_full, normal_quality_level1, normal_quality_level2, normal_quality_level3],
            normal_quality_full.shape[-2:],
        )
        geometry_quality_final = self._blend_multiscale(
            [geometry_quality_full, geometry_quality_level1, geometry_quality_level2, geometry_quality_level3],
            geometry_quality_full.shape[-2:],
        )
        prior_quality_final = torch.clamp((0.40 * semantic_quality_final) + (0.60 * geometry_quality_final), 0.0, 1.0)
         
        decoder_context = torch.cat([hv_1, i_dec1], dim=1)
        base_residual = torch.cat([hv_0, i_dec0], dim=1)
        observable_features = torch.cat([decoder_context, obs_final], dim=1)
        observable_delta_base = self.observable_head(observable_features)
        observable_delta = observable_delta_base + self.observable_refiner(
            torch.cat([observable_features, observable_delta_base], dim=1)
        )
        semantic_support = torch.clamp(
            ((0.45 * rs_final) + (0.20 * semantic_agreement_full) + (0.30 * semantic_quality_final) + (0.05 * semantic_valid_full)) * semantic_valid_full,
            0.0,
            1.0,
        )
        geometry_support = torch.clamp(
            ((0.40 * rg_final) + (0.20 * geometry_agreement_full) + (0.30 * geometry_quality_final) + (0.10 * geometry_valid_full)) * geometry_valid_full,
            0.0,
            1.0,
        )
        support_consensus = torch.clamp(1.0 - torch.abs(semantic_support - geometry_support), 0.0, 1.0)
        prior_support_base = torch.clamp(
            torch.maximum(semantic_support, geometry_support) *
            ((0.65) + (0.35 * support_consensus)) *
            ((0.60) + (0.40 * prior_quality_final)),
            0.0,
            1.0,
        )
        prior_support_learned = self.prior_support_fusion(torch.cat([
            semantic_support,
            geometry_support,
            support_consensus,
            prior_quality_final,
            semantic_quality_final,
            geometry_quality_final,
            1.0 - obs_final,
        ], dim=1))
        prior_support = torch.clamp(
            (0.55 * prior_support_base) + (0.45 * prior_support_learned),
            0.0,
            1.0,
        )
        unobservable_features = torch.cat([
            decoder_context,
            semantic_feats['full'] * semantic_quality_final,
            geometry_feats_full * geometry_quality_final,
            1.0 - obs_final,
            semantic_support,
            geometry_support,
            semantic_quality_final,
            geometry_quality_final,
            prior_support,
        ], dim=1)
        raw_unobservable_delta_base = self.unobservable_head(unobservable_features)
        raw_unobservable_delta = raw_unobservable_delta_base + self.unobservable_refiner(
            torch.cat([unobservable_features, raw_unobservable_delta_base], dim=1)
        )
        color_completion_gate = torch.clamp(
            (0.55 * semantic_support) + (0.20 * geometry_support) + (0.10 * obs_final),
            0.0,
            0.45,
        ) * ((0.55) + (0.45 * support_consensus)) * completion_scale
        intensity_completion_gate = torch.clamp(
            (0.60 * geometry_support) + (0.25 * semantic_support) + (0.10 * prior_support),
            0.0,
            1.0,
        ) * ((0.70) + (0.30 * support_consensus)) * completion_scale
        unobservable_delta = torch.cat([
            raw_unobservable_delta[:, 0:2, :, :] * color_completion_gate,
            raw_unobservable_delta[:, 2:3, :, :] * intensity_completion_gate,
        ], dim=1)

        observable_hvi = hvi + base_residual + observable_delta
        unobservable_hvi = hvi + base_residual + unobservable_delta
        output_hvi = hvi + base_residual + (obs_final * observable_delta) + ((1.0 - obs_final) * unobservable_delta)
        output_rgb_base_raw = self.trans.PHVIT(output_hvi)
        observable_rgb = self.trans.PHVIT(observable_hvi)
        unobservable_rgb = self.trans.PHVIT(unobservable_hvi)
        refine_context = torch.cat([
            output_rgb_base_raw,
            x,
            obs_final,
            semantic_support,
            geometry_support,
            semantic_quality_final,
            geometry_quality_final,
            prior_support,
        ], dim=1)
        exposure_params = self.exposure_head(refine_context)
        exposure_scale = 1.0 + (0.12 * torch.tanh(exposure_params[:, 0:3, :, :]))
        exposure_bias = 0.06 * torch.tanh(exposure_params[:, 3:6, :, :])
        output_rgb_base = torch.clamp((output_rgb_base_raw * exposure_scale) + exposure_bias, 0.0, 1.0)
        rgb_refine_delta = self.rgb_refiner(torch.cat([
            output_rgb_base,
            x,
            obs_final,
            semantic_support,
            geometry_support,
            semantic_quality_final,
            geometry_quality_final,
            prior_support,
        ], dim=1))
        output_rgb = torch.clamp(output_rgb_base + rgb_refine_delta, 0.0, 1.0)

        if return_aux:
            return {
                'rgb': output_rgb,
                'hvi': output_hvi,
                'rgb_base': output_rgb_base,
                'rgb_base_raw': output_rgb_base_raw,
                'rgb_obs': observable_rgb,
                'rgb_unobs': unobservable_rgb,
                'hvi_obs': observable_hvi,
                'hvi_unobs': unobservable_hvi,
                'observable_delta_base': observable_delta_base,
                'observable_delta': observable_delta,
                'raw_unobservable_delta_base': raw_unobservable_delta_base,
                'raw_unobservable_delta': raw_unobservable_delta,
                'rgb_refine_delta': rgb_refine_delta,
                'obs': obs_final,
                'rs': rs_final,
                'rg': rg_final,
                'obs_pyramid': [obs_full, obs_level1, obs_level2, obs_level3],
                'rs_pyramid': [rs_full, rs_level1, rs_level2, rs_level3],
                'rg_pyramid': [rg_full, rg_level1, rg_level2, rg_level3],
                'semantic_prior': semantic_prior,
                'geometry_prior': geometry_prior,
                'depth_prior': depth_prior,
                'normal_prior': normal_prior,
                'semantic_quality_prior': semantic_quality_prior,
                'depth_quality_prior': depth_quality_prior,
                'normal_quality_prior': normal_quality_prior,
                'semantic_valid': semantic_valid,
                'depth_valid': depth_valid,
                'normal_valid': normal_valid,
                'geometry_valid': geometry_valid,
                'semantic_agreement': semantic_agreement_full,
                'depth_agreement': depth_agreement_full,
                'normal_agreement': normal_agreement_full,
                'geometry_agreement': geometry_agreement_full,
                'semantic_agreement_pyramid': [
                    semantic_agreement_full,
                    semantic_agreement_level1,
                    semantic_agreement_level2,
                    semantic_agreement_level3,
                ],
                'depth_agreement_pyramid': [
                    depth_agreement_full,
                    depth_agreement_level1,
                    depth_agreement_level2,
                    depth_agreement_level3,
                ],
                'normal_agreement_pyramid': [
                    normal_agreement_full,
                    normal_agreement_level1,
                    normal_agreement_level2,
                    normal_agreement_level3,
                ],
                'geometry_agreement_pyramid': [
                    geometry_agreement_full,
                    geometry_agreement_level1,
                    geometry_agreement_level2,
                    geometry_agreement_level3,
                ],
                'semantic_quality': semantic_quality_final,
                'depth_quality': depth_quality_final,
                'normal_quality': normal_quality_final,
                'geometry_quality': geometry_quality_final,
                'prior_quality': prior_quality_final,
                'semantic_quality_pyramid': [
                    semantic_quality_full,
                    semantic_quality_level1,
                    semantic_quality_level2,
                    semantic_quality_level3,
                ],
                'depth_quality_pyramid': [
                    depth_quality_full,
                    depth_quality_level1,
                    depth_quality_level2,
                    depth_quality_level3,
                ],
                'normal_quality_pyramid': [
                    normal_quality_full,
                    normal_quality_level1,
                    normal_quality_level2,
                    normal_quality_level3,
                ],
                'geometry_quality_pyramid': [
                    geometry_quality_full,
                    geometry_quality_level1,
                    geometry_quality_level2,
                    geometry_quality_level3,
                ],
                'semantic_support': semantic_support,
                'geometry_support': geometry_support,
                'support_consensus': support_consensus,
                'prior_support_base': prior_support_base,
                'prior_support_learned': prior_support_learned,
                'prior_support': prior_support,
                'completion_color_gate': color_completion_gate,
                'completion_intensity_gate': intensity_completion_gate,
                'completion_scale': torch.full_like(obs_final, completion_scale),
                'exposure_scale': exposure_scale,
                'exposure_bias': exposure_bias,
                'route_weights': [
                    route_i1, route_h1,
                    route_i2, route_h2,
                    route_i3, route_h3,
                    route_i4, route_h4,
                    route_i5, route_h5,
                    route_i6, route_h6,
                ],
                'route_strengths': [
                    strength_i1, strength_h1,
                    strength_i2, strength_h2,
                    strength_i3, strength_h3,
                    strength_i4, strength_h4,
                    strength_i5, strength_h5,
                    strength_i6, strength_h6,
                ],
            }

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    
    
