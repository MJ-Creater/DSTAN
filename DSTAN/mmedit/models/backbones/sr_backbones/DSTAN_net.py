import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from torchvision import models
# from CAB_model.merge_net import MergeNet

# from Basicsr.models.archs.vsrTransformer_arch import Transformer
# from positional_encodings.torch_encodings import PositionalEncodingPermute3D
import einops
import matplotlib.pyplot as plt
import cv2

@BACKBONES.register_module()
class COLA_TTVSRNet(nn.Module):
    """COLA_TTVSR

    Support only x4 upsampling.
    Paper:
        Learning Trajectory-Aware Transformer for Video Super-Resolution, CVPR, 2022

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in propagation branch.
            Default: 60.
        stride (int): the scale of tokens.
            Default: 4.
        frame_stride (int): Number determining the stride of frames. If frame_stride=3,
            then the (0, 3, 6, 9, ...)-th frame will be the slected frames.
            Default: 3.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, mid_channels=64, num_blocks=60, stride=4, frame_stride=3,spynet_pretrained=None, num_frame=10):

        super().__init__()

        self.mid_channels = mid_channels
        self.keyframe_stride = frame_stride
        self.stride = stride
        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # Transformer
        # self.pos_embedding = PositionalEncodingPermute3D(num_frame)
        # self.transformer = Transformer(num_feat=64, feat_size=64, depth=7, patch_size=8, heads=1)

        # 空间协作注意力降噪（针对每帧）
        # self.cola_net = MergeNet(in_channels=3, intermediate_channels=64, vector_length=32, use_multiple_size=True, dncnn_depth=6,
        #                num_merge_block=4)
        self.feat_extractor = ResidualBlocksWithInputConv(
            3, mid_channels, 5)

        # self.LTAM = LTAM(stride = self.stride)
        # 可变形时空注意力
        self.DSTAM = DSTAM(stride=self.stride)

        # propagation branches
        self.resblocks = ResidualBlocksWithInputConv(
            2 * mid_channels, mid_channels, num_blocks)
        # upsample
        self.fusion = nn.Conv2d(
            3 * mid_channels, mid_channels, 1, 1, 0, bias=True)
        # self.upsample1 = PixelShufflePack(
        #     mid_channels, mid_channels, 2, upsample_kernel=3)
        # self.upsample2 = PixelShufflePack(
        #     mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        # self.img_upsample = nn.Upsample(
        #     scale_factor=4, mode='bilinear', align_corners=False)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, to_cpu=False):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        # n为batchsize t为短序列复制了两次（5*2*2=20）
        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        # compute optical flow
        # 光流用于预对齐和更新位图
        flows_forward, flows_backward = self.compute_flow(lrs)

        # 提取主干特征
        outputs = self.feat_extractor(lrs.view(-1, c, h, w)).view(n, t, -1, h, w)

        # 注意时间特征
        outputs = torch.unbind(outputs,dim=1)
        outputs = list(outputs)

        keyframe_idx_forward = list(range(0, t, self.keyframe_stride))
        keyframe_idx_backward = list(range(t-1, 0, 0-self.keyframe_stride))

        # backward-time propgation
        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h//self.stride), torch.arange(0, w//self.stride))
        location_update = torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)
        for i in range(t - 1, -1, -1):
            # lr_curr = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[i]
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1),padding_mode='border')

                # update the location map
                flow = F.adaptive_avg_pool2d(flow,(h//self.stride,w//self.stride))/self.stride
                location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")# n , 2t , h , w

                feat_buffer = torch.stack(feat_buffers, dim=1)
                sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                sparse_feat_buffer_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)
                sparse_feat_buffer_s3 = torch.stack(sparse_feat_buffers_s3, dim=1)
                index_feat_buffer_s1 = torch.stack(index_feat_buffers_s1, dim=1)
                # lr_curr_feat在LTAM里投影到Q，index_feat_buffer_s1投影到K，输入都是当前帧特征
                # sparse_feat输入是feat_prop（融合了参考帧之前或之后所有帧的信息，所谓特征传播），投影到V
                feat_prop = self.DSTAM(lr_curr_feat,index_feat_buffer_s1,feat_prop,sparse_feat_buffer_s1,sparse_feat_buffer_s2,sparse_feat_buffer_s3,location_update)

                # init the location map
                if i in keyframe_idx_backward:
                    location_update = torch.cat([location_update,torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)],dim=1) # n , 2t , h , w
            feat_prop = torch.cat([lr_curr_feat,feat_prop], dim=1)
            feat_prop = self.resblocks(feat_prop)
            feat_buffers.append(feat_prop)
            if i in keyframe_idx_backward:

                # feature tokenization *4
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s1 = F.unfold(feat_prop, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s1 = F.fold(sparse_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                index_feat_prop_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                index_feat_prop_s1 = F.fold(index_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
                index_feat_buffers_s1.append(index_feat_prop_s1)

                # feature tokenization *6
                # bs * c * h * w --> # bs * (c*6*6) * (h//4*w//4)
                sparse_feat_prop_s2 = F.unfold(feat_prop, kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=int(0.25*self.stride), stride=self.stride)
                # bs * (c*6*6) * (h//4*w//4) -->  bs * c * (h*1.5) * (w*1.5)
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(1.5*h),int(1.5*w)), kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=0, stride=int(1.5*self.stride))
                # bs * c * (h*1.5) * (w*1.5) -->  bs * c * h * w
                sparse_feat_prop_s2 = F.adaptive_avg_pool2d(sparse_feat_prop_s2,(h,w))
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s2 = F.unfold(sparse_feat_prop_s2, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s2.append(sparse_feat_prop_s2)

                # feature tokenization * 8
                # bs * c * h * w --> # bs * (c*8*8) * (h//4*w//4)
                sparse_feat_prop_s3 = F.unfold(feat_prop, kernel_size=(int(2*self.stride),int(2*self.stride)), padding=int(0.5*self.stride), stride=self.stride)
                # bs * (c*8*8) * (h//4*w//4) -->  bs * c * (h*2) * (w*2)
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(2*h),int(2*w)), kernel_size=(int(2*self.stride),int(2*self.stride)), padding=0, stride=int(2*self.stride))
                # bs * c * (h*2) * (w*2) -->  bs * c * h * w
                sparse_feat_prop_s3 = F.adaptive_avg_pool2d(sparse_feat_prop_s3,(h,w))
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s3 = F.unfold(sparse_feat_prop_s3, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s3.append(sparse_feat_prop_s3)

        outputs_back = feat_buffers[::-1]
        del location_update
        del feat_buffers
        del sparse_feat_buffers_s1
        del sparse_feat_buffers_s2
        del sparse_feat_buffers_s3
        del index_feat_buffers_s1

        # forward-time propagation and upsampling
        fina_out = []
        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []

        feat_prop = torch.zeros_like(feat_prop)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h//self.stride), torch.arange(0, w//self.stride))
        location_update = torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)
        for i in range(0, t):
            # 原：
            lr_curr = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[i]

            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1),padding_mode='border')

                # update the location map
                flow = F.adaptive_avg_pool2d(flow,(h//self.stride,w//self.stride))/self.stride
                location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")# n , 2t , h , w

                feat_buffer = torch.stack(feat_buffers, dim=1)
                sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                sparse_feat_buffer_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)
                sparse_feat_buffer_s3 = torch.stack(sparse_feat_buffers_s3, dim=1)
                index_feat_buffer_s1 = torch.stack(index_feat_buffers_s1, dim=1)
                feat_prop = self.DSTAM(lr_curr_feat,index_feat_buffer_s1,feat_prop,sparse_feat_buffer_s1,sparse_feat_buffer_s2,sparse_feat_buffer_s3,location_update)

                # init the location map
                if i in keyframe_idx_forward:
                    location_update = torch.cat([location_update,torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)],dim=1)
            feat_prop = torch.cat([outputs[i], feat_prop], dim=1)
            feat_prop = self.resblocks(feat_prop)
            feat_buffers.append(feat_prop)

            if i in keyframe_idx_forward:
                # feature tokenization *4
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s1 = F.unfold(feat_prop, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s1 = F.fold(sparse_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                index_feat_prop_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                index_feat_prop_s1 = F.fold(index_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
                index_feat_buffers_s1.append(index_feat_prop_s1)


                # feature tokenization *6
                # bs * c * h * w --> # bs * (c*6*6) * (h//4*w//4)
                sparse_feat_prop_s2 = F.unfold(feat_prop, kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=int(0.25*self.stride), stride=self.stride)
                # bs * (c*6*6) * (h//4*w//4) -->  bs * c * (h*1.5) * (w*1.5)
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(1.5*h),int(1.5*w)), kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=0, stride=int(1.5*self.stride))
                # bs * c * (h*1.5) * (w*1.5) -->  bs * c * h * w
                sparse_feat_prop_s2 = F.adaptive_avg_pool2d(sparse_feat_prop_s2,(h,w))
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s2 = F.unfold(sparse_feat_prop_s2, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s2.append(sparse_feat_prop_s2)


                # feature tokenization *8
                # bs * c * h * w --> # bs * (c*8*8) * (h//4*w//4)
                sparse_feat_prop_s3 = F.unfold(feat_prop, kernel_size=(int(2*self.stride),int(2*self.stride)), padding=int(0.5*self.stride), stride=self.stride)
                # bs * (c*8*8) * (h//4*w//4) -->  bs * c * (h*2) * (w*2)
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(2*h),int(2*w)), kernel_size=(int(2*self.stride),int(2*self.stride)), padding=0, stride=int(2*self.stride))
                # bs * c * (h*2) * (w*2) -->  bs * c * h * w
                sparse_feat_prop_s3 = F.adaptive_avg_pool2d(sparse_feat_prop_s3,(h,w))
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s3 = F.unfold(sparse_feat_prop_s3, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
                # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s3.append(sparse_feat_prop_s3)

            # upsampling given the backward and forward features
            out = torch.cat([outputs_back[i],lr_curr_feat,feat_prop], dim=1)
            # 再并联上空间注意力降噪的结果
            # out = torch.cat([outputs_back[i], lr_curr_feat, feat_prop, lr_curr_spatial_feat], dim=1)

            out = self.lrelu(self.fusion(out))
            # out = self.lrelu(self.upsample1(out))
            # out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            # 原图上采样
            # base = self.img_upsample(lr_curr)
            base = lr_curr
            out += base
            if to_cpu:
                fina_out.append(out.cpu())
            else:
                fina_out.append(out)
        del location_update
        del sparse_feat_buffers_s1
        del sparse_feat_buffers_s2
        del sparse_feat_buffers_s3
        del index_feat_buffers_s1
        return torch.stack(fina_out, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class LTAM(nn.Module):
    def __init__(self, stride=4):
        super().__init__()

        self.stride = stride
        self.fusion = nn.Conv2d(3 * 64, 64, 3, 1, 1, bias=True)
    def forward(self, curr_feat, index_feat_set_s1 , anchor_feat, sparse_feat_set_s1 ,sparse_feat_set_s2, sparse_feat_set_s3, location_feat):
        """Compute the long-range trajectory-aware attention.

        Args:
            anchor_feat (tensor): Input feature with shape (n, c, h, w)
            sparse_feat_set_s1 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s2 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s3 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            location_feat (tensor): Input location map with shape (n, 2*t, h//4, w//4)
 
        Return:
            fusion_feature (tensor): Output fusion feature with shape (n, c, h, w).
        """

        n, c, h, w = anchor_feat.size()
        t = sparse_feat_set_s1.size(1)
        feat_len = int(c*self.stride*self.stride)
        feat_num = int((h//self.stride) * (w//self.stride))

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        grid_flow = location_feat.contiguous().view(n,t,2,h//self.stride,w//self.stride).permute(0, 1, 3, 4, 2)
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w//self.stride - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h//self.stride - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)

        output_s1 = F.grid_sample(sparse_feat_set_s1.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s2 = F.grid_sample(sparse_feat_set_s2.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s3 = F.grid_sample(sparse_feat_set_s3.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
     
        index_output_s1 = F.grid_sample(index_feat_set_s1.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        # n * c * h * w --> # n * (c*4*4) * (h//4*w//4)
        curr_feat = F.unfold(curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
        # n * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * (c*4*4)
        curr_feat = curr_feat.permute(0, 2, 1)
        # 这里完了就相当于Q
        curr_feat = F.normalize(curr_feat, dim=2).unsqueeze(3) # n * (h//4*w//4) * (c*4*4) * 1

        # cross-scale attention * 4
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        index_output_s1 = index_output_s1.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        index_output_s1 = F.unfold(index_output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * t * (c*4*4)
        index_output_s1 = index_output_s1.permute(0, 3, 1, 2)
        index_output_s1 = F.normalize(index_output_s1, dim=3) # n * (h//4*w//4) * t * (c*4*4)
        # [ n * (h//4*w//4) * t * (c*4*4) ]  *  [ n * (h//4*w//4) * (c*4*4) * 1 ]  -->  n * (h//4*w//4) * t
        matrix_index = torch.matmul(index_output_s1, curr_feat).squeeze(3) # n * (h//4*w//4) * t
        matrix_index = matrix_index.view(n,feat_num,t)# n * (h//4*w//4) * t
        corr_soft, corr_index = torch.max(matrix_index, dim=2)# n * (h//4*w//4)
        # n * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        corr_soft = corr_soft.unsqueeze(1).expand(-1,feat_len,-1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        corr_soft = F.fold(corr_soft, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s1 = output_s1.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s1 = F.unfold(output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s1 = torch.gather(output_s1.contiguous().view(n,t,feat_len,feat_num), 1, corr_index.view(n,1,1,feat_num).expand(-1,-1,feat_len,-1))# n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4)  --> n * (c*4*4) * (h//4*w//4)
        output_s1 = output_s1.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s1 = F.fold(output_s1, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        # Aggr
         # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s2 = output_s2.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s2 = F.unfold(output_s2, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)  
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)     
        output_s2 = torch.gather(output_s2.contiguous().view(n,t,feat_len,feat_num), 1, corr_index.view(n,1,1,feat_num).expand(-1,-1,feat_len,-1))# n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        output_s2 = output_s2.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s2 = F.fold(output_s2, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s3 = output_s3.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s3 = F.unfold(output_s3, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)  
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)  
        output_s3 = torch.gather(output_s3.contiguous().view(n,t,feat_len,feat_num), 1, corr_index.view(n,1,1,feat_num).expand(-1,-1,feat_len,-1))# n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        output_s3 = output_s3.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s3 = F.fold(output_s3, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        out = torch.cat([output_s1,output_s2,output_s3], dim=1)
        out = self.fusion(out)
        out = out * corr_soft
        out += anchor_feat
        return out

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class DSTAM(nn.Module):
    """Deformable Spatio-Temporal Attention Module

    """
    def __init__(self, stride=4):
        super().__init__()

        self.stride = stride
        self.fusion = nn.Conv2d(3 * 64, 64, 3, 1, 1, bias=True)
        self.channels = 1024    # 投影到空间QVK后，特征向量的长度
        self.scale = self.channels ** -0.5
        self.n_groups = 4
        self.n_group_channels = self.channels // self.n_groups
        self.time_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels * 2, self.n_group_channels, 5, 1, 5 // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        self.space_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, 5, 1, 5 // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        self.offset_range_factor = 2
        # 原始特征的通道数为64
        self.nc = 64
        self.proj_q = nn.Conv2d(
            self.channels, self.channels,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_k = nn.Conv2d(
            self.channels, self.channels,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, curr_feat, index_feat_set_s1, anchor_feat, sparse_feat_set_s1, sparse_feat_set_s2,
                sparse_feat_set_s3, location_feat):
        """Compute the long-range trajectory-aware attention.

        Args:
            anchor_feat (tensor): Input feature with shape (n, c, h, w)
            sparse_feat_set_s1 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s2 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s3 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            location_feat (tensor): Input location map with shape (n, 2*t, h//4, w//4)

        Return:
            fusion_feature (tensor): Output fusion feature with shape (n, c, h, w).
        """

        n, c, h, w = anchor_feat.size()
        dtype, device = anchor_feat.dtype, anchor_feat.device
        #
        t = sparse_feat_set_s1.size(1)
        feat_len = int(c * self.stride * self.stride)
        feat_num = int((h // self.stride) * (w // self.stride))

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        grid_flow = location_feat.contiguous().view(n, t, 2, h // self.stride, w // self.stride).permute(0, 1, 3, 4, 2)
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w // self.stride - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h // self.stride - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)

        output_s1 = F.grid_sample(
            sparse_feat_set_s1.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                 (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s2 = F.grid_sample(
            sparse_feat_set_s2.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                 (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s3 = F.grid_sample(
            sparse_feat_set_s3.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                 (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)

        # 时间注意力的K每次扩张于关键帧，如果每帧都计算K会导致大的计算量并且相邻几帧运动很小，所以每帧使用位图调整关键帧K的位置即可
        # 但是当几帧过后，运动较大，只调整原来关键帧的k不够用了，所以需要扩张新的关键帧的K
        # 用位图调整完的K就相当于其他支持帧的K，而Q就来自于当前参考帧，Q总领先于K
        index_output_s1 = F.grid_sample(
            index_feat_set_s1.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)

        # cross-scale attention * 4
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        index_output_s1 = index_output_s1.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                            (w // self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        index_output_s1 = F.unfold(index_output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len,
                                                                                                  feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * t * (c*4*4)
        index_output_s1 = index_output_s1.permute(0, 3, 1, 2)
        index_output_s1 = F.normalize(index_output_s1, dim=3)  # n * (h//4*w//4) * t * (c*4*4)

        # Q的产生
        # n * c * h * w --> # n * (c*4*4) * (h//4*w//4)
        curr_feat = F.unfold(curr_feat, kernel_size=(self.stride, self.stride), padding=0, stride=self.stride)

        # spatial q和k本质上都来自于同一当前帧，经过不同的卷积投影
        spatial_q = self.proj_q(curr_feat.reshape(-1, self.channels, (h // self.stride), (w // self.stride)))
        spatial_k = self.proj_k(curr_feat.reshape(-1, self.channels, (h // self.stride), (w // self.stride)))

        # n * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * (c*4*4)
        curr_feat = curr_feat.permute(0, 2, 1)
        # 这里完了就相当于Q
        curr_feat = F.normalize(curr_feat, dim=2).unsqueeze(3)  # n * (h//4*w//4) * (c*4*4) * 1
        
        # **********************空间变形注意约束时间注意的位置********************************************
        temporal_q = curr_feat.squeeze(3).contiguous().view(-1, (h // self.stride),(w // self.stride),
                                                self.channels).permute(0,3,1,2)
        # n * (h//4*w//4) * t * (c*4*4) -> n*g,Cg,Hg,Wg
        temporal_k = index_output_s1.permute(0, 2, 1, 3).contiguous().view(-1, (h // self.stride), (w // self.stride),
                                                                           self.channels).permute(0, 3, 1, 2)
        
        
        # 在channels上分为四组，每组学习不同的变形点并组内共享
        q_off = einops.rearrange(temporal_q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        # 时间q_off与k_off batch保持一致
        q_off = q_off.repeat(t, 1, 1, 1)
        k_off = einops.rearrange(temporal_k, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        off = torch.cat([q_off, k_off], dim=1)
        offset = self.time_offset(off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        
        # 偏移量幅值缩放，防止偏移过大
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # 生成采样点
        reference = self._get_ref_points(Hk, Wk, n * t, dtype, device)
        # 将采样点偏移, pos为变形点
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        
        # spatial_k分四组偏移
        temporal_k_sampled = F.grid_sample(
            input=temporal_k.reshape(n * t * self.n_groups, self.n_group_channels, Hk, Wk),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        # n_sample这里是支持帧变形过位置的k (n,s,t,c)
        temporal_k_sampled = einops.rearrange(temporal_k_sampled, '(n t g) c h w -> n (h w) t (g c)', t=t, g=self.n_groups)
        
        # [ n * (h//4*w//4) * t * (c*4*4) ]  *  [ n * (h//4*w//4) * (c*4*4) * 1 ]  -->  n * (h//4*w//4) * t
        # 这里注意力改成是参考帧与支持帧变形过后的时间注意
        matrix_index = torch.matmul(temporal_k_sampled, curr_feat).squeeze(3)  # n * (h//4*w//4) * t
        matrix_index = matrix_index.view(n, feat_num, t)  # n * (h//4*w//4) * t
        corr_soft, corr_index = torch.max(matrix_index, dim=2)  # n * (h//4*w//4)
        # n * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        corr_soft = corr_soft.unsqueeze(1).expand(-1, feat_len, -1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        corr_soft = F.fold(corr_soft, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)
        
        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s1 = output_s1.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride))
        
        temporal_v_sampled = F.grid_sample(
            input=output_s1.reshape(n * t * self.n_groups, self.n_group_channels, (h // self.stride),
                                    (w // self.stride)),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        temporal_v_sampled = temporal_v_sampled.reshape(-1, self.channels, Hk, Wk)
        
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s1 = F.unfold(temporal_v_sampled, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s1 = torch.gather(output_s1.contiguous().view(n, t, feat_len, feat_num), 1,
                                 corr_index.view(n, 1, 1, feat_num).expand(-1, -1, feat_len,
                                                                           -1))  # n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4)  --> n * (c*4*4) * (h//4*w//4)
        output_s1 = output_s1.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s1 = F.fold(output_s1, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)
        
        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s2 = output_s2.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride))
        
        temporal_v_sampled = F.grid_sample(
            input=output_s2.reshape(n * t * self.n_groups, self.n_group_channels, (h // self.stride),
                                    (w // self.stride)),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        temporal_v_sampled = temporal_v_sampled.reshape(-1, self.channels, Hk, Wk)
        
        
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s2 = F.unfold(temporal_v_sampled, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s2 = torch.gather(output_s2.contiguous().view(n, t, feat_len, feat_num), 1,
                                 corr_index.view(n, 1, 1, feat_num).expand(-1, -1, feat_len,
                                                                           -1))  # n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        output_s2 = output_s2.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s2 = F.fold(output_s2, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)
        
        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s3 = output_s3.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride))
        
        temporal_v_sampled = F.grid_sample(
            input=output_s3.reshape(n * t * self.n_groups, self.n_group_channels, (h // self.stride),
                                    (w // self.stride)),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        temporal_v_sampled = temporal_v_sampled.reshape(-1, self.channels, Hk, Wk)
        
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s3 = F.unfold(temporal_v_sampled, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s3 = torch.gather(output_s3.contiguous().view(n, t, feat_len, feat_num), 1,
                                 corr_index.view(n, 1, 1, feat_num).expand(-1, -1, feat_len,
                                                                           -1))  # n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        output_s3 = output_s3.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s3 = F.fold(output_s3, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)
        
        temporal_out = torch.cat([output_s1, output_s2, output_s3], dim=1)
        temporal_out = self.fusion(temporal_out)
        # 时间注意力（融合完的）结果
        temporal_out = temporal_out * corr_soft

        # # --------------------------------------纯DSAM模块（消融实验）---------------------------------------------

        # # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        # output_s1 = output_s1.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
        #                                         (w // self.stride))
        # # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        # output_s1 = F.unfold(output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # # n * t * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        # output_s1 = output_s1[:, -1] # n * (c*4*4) * (h//4*w//4)
        # # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        # output_s1 = F.fold(output_s1, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
        #                    stride=self.stride)

        # # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        # output_s2 = output_s2.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
        #                                         (w // self.stride))
        # # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        # output_s2 = F.unfold(output_s2, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # # n * t * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        # output_s2 = output_s2[:, -1]  # n * (c*4*4) * (h//4*w//4)
        # # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        # output_s2 = F.fold(output_s2, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
        #                    stride=self.stride)

        # # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        # output_s3 = output_s3.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
        #                                         (w // self.stride))
        # # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        # output_s3 = F.unfold(output_s3, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # # n * t * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        # output_s3 = output_s3[:, -1]  # n * (c*4*4) * (h//4*w//4)
        # # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        # output_s3 = F.fold(output_s3, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
        #                    stride=self.stride)
        # temporal_out = torch.cat([output_s1, output_s2, output_s3], dim=1)
        # temporal_out = self.fusion(temporal_out)

        # # --------------------------------------纯DSAM模块（消融实验）---------------------------------------------

        # DSAM-空间变形注意力
        # 在channels上分为四组，每组学习不同的变形点并组内共享
        q_off = einops.rearrange(spatial_q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.space_offset(q_off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        # 偏移量幅值缩放，防止偏移过大
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # 生成采样点
        reference = self._get_ref_points(Hk, Wk, n, dtype, device)


        # # ----------------------------参考点可视化-----------------------
        # ref_points = copy.deepcopy(reference[1])
        # ref_points[..., 0].add_(1).div_(2).mul_(Hk)
        # ref_points[..., 1].add_(1).div_(2).mul_(Wk)
        #
        # points_x, points_y = ref_points[0, ::5, 1], ref_points[::5, 0, 0]
        # X, Y = torch.meshgrid(points_x, points_y)
        # X, Y = X.cpu().numpy(), Y.cpu().numpy()
        # plt.plot(X, Y,
        #          color='red',  # 全部点设置为红色
        #          marker='.',  # 点的形状为圆点
        #          linestyle='')  # 线型为空，也即点与点之间不用线连接
        # # plt.grid(True)
        # plt.show()
        # # ----------------------------参考点可视化-----------------------


        # 将采样点偏移, pos为变形点
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        # # ----------------------------变形点可视化-----------------------
        # deformed_points = copy.deepcopy(pos[1])
        # deformed_points[..., 0].add_(1).div_(2).mul_(Hk)
        # deformed_points[..., 1].add_(1).div_(2).mul_(Wk)
        #
        # points_x, points_y = deformed_points[0, ::5, 1], deformed_points[::5, 0, 0]
        # X, Y = torch.meshgrid(points_x, points_y)
        # X, Y = X.cpu().numpy(), Y.cpu().numpy()
        # plt.plot(X, Y,
        #          color='red',  # 全部点设置为红色
        #          marker='.',  # 点的形状为圆点
        #          linestyle='')  # 线型为空，也即点与点之间不用线连接
        # # plt.grid(True)
        # plt.show()
        # # ----------------------------变形点可视化-----------------------


        spatial_k_sampled = F.grid_sample(
            input=spatial_k.reshape(n * self.n_groups, self.n_group_channels, Hk, Wk),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        spatial_q = spatial_q.reshape(-1, self.channels, n_sample).permute(0, 2, 1)
        spatial_k_sampled = spatial_k_sampled.reshape(-1, self.channels, n_sample)

        attn = torch.matmul(spatial_q, spatial_k_sampled)
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=-1)

        # # ---------------------------注意力可视化,第(x, y)个query点----------------------------------
        # axis_x, axis_y = 75, 100
        # attn_map = attn[0, axis_y*Wk+axis_x].reshape(Hk, Wk).cpu().numpy()
        # plt.imshow(attn_map)
        # plt.scatter(axis_x, axis_y, c='r', s=100, marker='o')
        # plt.show()
        # # ---------------------------注意力可视化,第i个query点----------------------------------


        spatial_v = self.proj_v(temporal_out)
        spatial_v = F.unfold(spatial_v, kernel_size=(self.stride, self.stride), padding=0, stride=self.stride)
        spatial_v = spatial_v.contiguous().view(-1, self.channels, Hk, Wk)
        # spatial_v分四组偏移
        spatial_v_sampled = F.grid_sample(
            input=spatial_v.reshape(n * self.n_groups, self.n_group_channels, Hk, Wk),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        # (n,256,1024)
        spatial_v_sampled = spatial_v_sampled.reshape(-1, self.channels, n_sample).permute(0, 2, 1)

        out = torch.matmul(attn, spatial_v_sampled).transpose(2, 1)
        spatio_temporal_out = F.fold(out, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)

        # fig = spatio_temporal_out.cpu().numpy()[0, 0]
        # plt.imshow(fig)
        # plt.show()

        spatio_temporal_out = self.proj_out(spatio_temporal_out)
        spatio_temporal_out += anchor_feat

        # temporal_out += anchor_feat
        return spatio_temporal_out
        # return temporal_out


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output
