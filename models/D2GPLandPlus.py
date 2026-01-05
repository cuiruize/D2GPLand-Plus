# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, cat, Tensor

from src.models.resnet import ResNet18, ResNet34, ResNet50
from src.models.context_modules import get_context_module
from src.models.model_utils import ConvBNAct, Swish, Hswish, \
    SqueezeAndExcitation
from src.models.model import Decoder
from segment_anything import sam_model_registry
from depth_anything_v2.dpt import DepthAnythingV2
from models.boundary_detector import Boundary_Detector
from models.CUMamba import CrossMamba


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class Granularity(nn.Module):
  def __init__(self):
    super().__init__()
    self.graularity = nn.Parameter(data=torch.zeros([3, 1]), requires_grad=True)

  def forward(self):
    temp = torch.sigmoid(self.graularity)
    return temp


class Prompt_Fusion(nn.Module):

    def __init__(self):
        super(Prompt_Fusion, self).__init__()

        self.FFN = nn.Sequential(
            nn.Conv2d(512 * 3, 512 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512 * 2),
            nn.ReLU(),
            nn.Conv2d(512 * 2, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.ablation_FFN = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.key_projections = nn.ModuleList()
        self.edge_semantics = nn.ModuleList()
        self.prompt_feat_projections = nn.ModuleList()
        self.fusion_forwards = nn.ModuleList()
        self.boundary_detectors = nn.ModuleList()

        for i in range(3):
            self.key_projections.append(nn.Linear(256, 256))
            self.fusion_forwards.append(nn.Sequential(
                nn.Conv2d(512 + 256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ))
            self.edge_semantics.append(nn.Sequential(
                # ablation study 去掉
                nn.Conv2d(512 + 512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)
            ))
            self.boundary_detectors.append(Boundary_Detector(512, 256))
            self.prompt_feat_projections.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ))

        self.granularity_coe = nn.Parameter(torch.zeros(3,1), requires_grad=True)

    def granularity_weighting(self, img_fft, alpha):
        ratio = 0.3
        batch_size, h, w, c = img_fft.shape
        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
        img_abs = torch.fft.fftshift(img_abs, dim=(1))
        h_crop = int(h * math.sqrt(ratio))
        w_crop = int(w * math.sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = 0
        img_abs_ = img_abs.clone()
        masks = torch.ones_like(img_abs)
        masks = masks * alpha
        masks[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = 1
        img_abs = img_abs_ * masks
        img_abs = torch.fft.ifftshift(img_abs, dim=(1))  # recover
        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def forward(self, in_features, sam_feature, prototypes):
        fused_feat = None
        edge_map = None

        features = in_features.reshape(in_features.shape[0], -1,
                                       in_features.shape[2] * in_features.shape[3])  # 2,512,256
        prior_prototypes = prototypes.unsqueeze(-1)
        sam_feat = F.interpolate(sam_feature, size=16, mode='bilinear').reshape(sam_feature.shape[0],
                                                                                -1,
                                                                                16 * 16)  # 2, 256, 256

        for i in range(prior_prototypes.shape[0]):
            specific_prototype = prior_prototypes[i]
            specific_prototype = torch.stack([specific_prototype for _ in range(sam_feature.shape[0])], dim=0)
            sim = torch.matmul(sam_feat, specific_prototype).squeeze(-1)
            sim = torch.stack([sim for _ in range(sam_feat.shape[1])], dim=1)
            specific_sam_feat = sam_feat + sam_feat * sim  # 2, 256, (16*16)
            specific_sam_feat = self.prompt_feat_projections[i](
                specific_sam_feat.reshape(specific_sam_feat.shape[0], -1, 16, 16)).reshape(specific_sam_feat.shape[0],
                                                                                           -1, 256)  # 2,256,16,16

            # fuse
            fused_feature = self.fusion_forwards[i](
                torch.cat((features, specific_sam_feat), dim=1).reshape(specific_sam_feat.shape[0], -1, 16, 16))


            # Gabor
            edge_attn = self.boundary_detectors[i](fused_feature) * self.granularity_coe[i]

            b, C, h, w = fused_feature.shape
            img_fft = fused_feature.permute(0, 2, 3, 1)
            img_fft = torch.fft.rfft2(img_fft, dim=(1, 2), norm='ortho')
            img_fft = self.granularity_weighting(img_fft, self.granularity_coe[i])
            img_fft = torch.fft.irfft2(img_fft, s=(h, w), dim=(1, 2), norm='ortho')
            img_fft = img_fft.permute(0, 3, 1, 2)

            map = self.edge_semantics[i](torch.cat([edge_attn, img_fft], 1))


            if fused_feat is None:
                fused_feat = fused_feature
                edge_map = map
            else:
                fused_feat = torch.cat((fused_feature, fused_feat), dim=1)
                edge_map = torch.cat((edge_map, map), dim=1)

        out = self.FFN(fused_feat)

        return out, edge_map, specific_sam_feat.reshape(specific_sam_feat.shape[0], -1, 16, 16)


class ESANetOneModality_sam(nn.Module):
    def __init__(self,
                 height=256,
                 width=480,
                 num_classes=4,
                 encoder='resnet34',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='/results_nas/moko3016/'
                                'moko3016-efficient-rgbd-segmentation/'
                                'imagenet_pretraining',
                 activation='relu',
                 input_channels=3,
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 weighting_in_encoder='None',
                 upsampling='bilinear'):
        super(ESANetOneModality_sam, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [3, 3, 3]

        self.weighting_in_encoder = weighting_in_encoder

        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError('Only relu, swish and hswish as '
                                      'activation function are supported so '
                                      'far. Got {}'.format(activation))
        sam = sam_model_registry["vit_b"](
            checkpoint="path/sam_vit_b_01ec64.pth")
        self.sam_encoder = sam.image_encoder
        depth_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        depth_encoder = 'vitb'
        self.depth_encoder = DepthAnythingV2(**depth_model_configs[depth_encoder])
        self.depth_encoder.load_state_dict(torch.load('path/depth_anything_v2_vitb.pth'))
        self.depth_encoder.eval()

        self.first_conv = ConvBNAct(1, 3, kernel_size=1, activation=self.activation)

        self.fusion_conv1 = ConvBNAct(768, 64, kernel_size=1, activation=self.activation)
        self.fusion_conv2 = ConvBNAct(768, 128, kernel_size=1, activation=self.activation)
        self.fusion_conv3 = ConvBNAct(768, 256, kernel_size=1, activation=self.activation)
        self.fusion_conv4 = ConvBNAct(256, 512, kernel_size=1, activation=self.activation)

        self.test_conv = nn.Conv2d(256, 512, kernel_size=1)
        self.test_conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.mid_conv = ConvBNAct(512 + 256, 512, kernel_size=1,
                                  activation=self.activation)

        # encoder
        if encoder == 'resnet18':
            self.encoder = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=input_channels
            )
        elif encoder == 'resnet34':
            self.encoder = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=input_channels
            )
        elif encoder == 'resnet50':
            self.encoder = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=input_channels
            )
        else:
            raise NotImplementedError('Only ResNets as encoder are supported '
                                      'so far. Got {}'.format(activation))

        self.channels_decoder_in = self.encoder.down_32_channels_out

        self.rgb_se = SqueezeAndExcitation(
            self.encoder.down_32_channels_out,
            activation=self.activation)
        self.depth_se = SqueezeAndExcitation(
            self.encoder.down_32_channels_out,
            activation=self.activation)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = get_context_module(
            context_module,
            self.channels_decoder_in,
            self.channels_decoder_in,
            input_size=(height // 32, width // 32),
            activation=self.activation,
            upsampling_mode=upsampling_context_module)

        self.pan_fusion1 = CrossMamba(self.encoder.down_32_channels_out, 32 * 32)
        self.pan_fusion2 = CrossMamba(self.encoder.down_32_channels_out, 32 * 32)
        self.pan_fusion3 = CrossMamba(self.encoder.down_32_channels_out, 32 * 32)
        self.pan_fusion4 = CrossMamba(self.encoder.down_32_channels_out, 32 * 32)
        self.pan_fusion5 = CrossMamba(self.encoder.down_32_channels_out, 32 * 32)

        # decoder
        self.decoder = Decoder(
            channels_in=channels_decoder[0],
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

        self.prompt_fusion_layer = Prompt_Fusion()

    def forward(self, image, prototypes, depth):
        original_depth = F.interpolate(image, size=(1022, 1022), mode='bilinear')
        original_depth = self.depth_encoder.infer_image(original_depth)

        original_depth = original_depth.expand(-1, 3, -1, -1)

        original_depth, layer_featrues = self.sam_encoder(original_depth)
        out = self.encoder.forward_first_conv(image)

        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        res = 0

        # block 1
        out1 = self.encoder.forward_layer1(out)
        skip1 = self.skip_layer1(out1)  # 32, 256, 256

        # block 2
        out2 = self.encoder.forward_layer2(out1)
        skip2 = self.skip_layer2(out2)  # 64, 128, 128

        # block 3
        out3 = self.encoder.forward_layer3(out2)
        skip3 = self.skip_layer3(out3)  # 128, 64, 64

        # block 4
        out4 = self.encoder.forward_layer4(out3)

        depth4 = self.fusion_conv4(
            F.interpolate(original_depth, size=(32, 32), mode='bilinear'))  # 256,64,64 -> 512,32,32
        out4, res = self.pan_fusion1(out4.view(out4.shape[0], -1, out4.shape[2] * out4.shape[3]).transpose(1, 2), res,
                                     depth4.view(depth4.shape[0], -1, depth4.shape[2] * depth4.shape[3]).transpose(1,
                                                                                                                   2),
                                     out4.shape[3], True)
        out4, res = self.pan_fusion2(out4.view(out4.shape[0], -1, out4.shape[2] * out4.shape[3]).transpose(1, 2), res,
                                     depth4.view(depth4.shape[0], -1, depth4.shape[2] * depth4.shape[3]).transpose(1,
                                                                                                                   2),
                                     out4.shape[3], True)
        out4, res = self.pan_fusion3(out4.view(out4.shape[0], -1, out4.shape[2] * out4.shape[3]).transpose(1, 2), res,
                                     depth4.view(depth4.shape[0], -1, depth4.shape[2] * depth4.shape[3]).transpose(1,
                                                                                                                   2),
                                     out4.shape[3], True)
        out4, res = self.pan_fusion4(out4.view(out4.shape[0], -1, out4.shape[2] * out4.shape[3]).transpose(1, 2), res,
                                     depth4.view(depth4.shape[0], -1, depth4.shape[2] * depth4.shape[3]).transpose(1,
                                                                                                                   2),
                                     out4.shape[3], True)
        out4, res = self.pan_fusion5(out4.view(out4.shape[0], -1, out4.shape[2] * out4.shape[3]).transpose(1, 2), res,
                                     depth4.view(depth4.shape[0], -1, depth4.shape[2] * depth4.shape[3]).transpose(1,
                                                                                                                   2),
                                     out4.shape[3], True)

        out = F.interpolate(out4, size=(16, 16), mode='bilinear')

        fused_feat, edge_out, geometry_feat = self.prompt_fusion_layer(out, original_depth, prototypes)
        fused_feat = F.interpolate(fused_feat, size=(32, 32), mode='bilinear')

        outs = [fused_feat, skip3, skip2, skip1]
        outs_decoder, out_cnn = self.decoder(enc_outs=outs)
        outs = F.log_softmax(outs_decoder, dim=1)

        return outs, original_depth, edge_out, out_cnn, outs_decoder
