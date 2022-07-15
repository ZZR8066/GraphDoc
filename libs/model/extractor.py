import torch
from torch import nn
from torch._C import device
from torchvision.ops import roi_align


def convert_to_roi_format(lines_box):
    concat_boxes = torch.cat(lines_box, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [
            torch.full((lines_box_pi.shape[0], 1), i, dtype=dtype, device=device)
            for i, lines_box_pi in enumerate(lines_box)
        ],
        dim=0
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


class RoIPool(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def gen_rois(self, feats):
        *_, H, W = feats.shape
        pool_W, pool_H = self.pool_size

        Width = W / pool_W
        Height = H / pool_H

        bbox_x = torch.arange(0, pool_W + 1, 1).to(feats) * Width
        bbox_y = torch.arange(0, pool_H + 1, 1).to(feats) * Height

        bboxes = torch.stack(
            [
                bbox_x[:-1].repeat(pool_W, 1),
                bbox_y[:-1].repeat(pool_H, 1).transpose(0, 1),
                bbox_x[1:].repeat(pool_W, 1),
                bbox_y[1:].repeat(pool_H, 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)
        
        rois = list()
        for batch_idx in range(feats.shape[0]):
            ids = torch.full((bboxes.shape[0], 1), batch_idx, dtype=feats.dtype, device=feats.device)
            rois.append(torch.cat([ids, bboxes], dim=-1))
        
        rois = torch.cat(rois, dim=0)
        return rois

    def forward(self, feats):
        rois = self.gen_rois(feats)
        bboxes_feat = roi_align(
            input=feats,
            boxes=rois,
            output_size=(1, 1),
            spatial_scale=1.0,
            sampling_ratio=1
        )
        bs = feats.shape[0]
        len = int(self.pool_size[0] * self.pool_size[1])
        bboxes_feat = bboxes_feat.reshape(bs, len, -1)
        return bboxes_feat


def tensor_convert_to_roi_format(line_bboxes):
    B, L, _ = line_bboxes.shape
    roi_ids = torch.zeros((B, L, 1)).to(line_bboxes).float()
    for id in range(B):
        roi_ids[id] = id
    rois = torch.cat([roi_ids, line_bboxes], dim=-1).reshape(-1, 5)
    return rois


class RoiFeatExtraxtor(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, feats, line_bboxes):
        rois = tensor_convert_to_roi_format(line_bboxes)
        lines_feat = roi_align(
            input=feats,
            boxes=rois,
            output_size=(1, 1),
            spatial_scale=self.scale,
            sampling_ratio=1
        )
        
        lines_feat = lines_feat.reshape(lines_feat.shape[0], -1)
        view_shape = line_bboxes.shape[:2]
        lines_feat = lines_feat.view(*view_shape,-1)
        return lines_feat


class RecogFeatExtraxtor(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, feats, line_bboxes, output_size=(1,1)):
        rois = tensor_convert_to_roi_format(line_bboxes)
        lines_feat = roi_align(
            input=feats,
            boxes=rois,
            output_size=output_size,
            spatial_scale=self.scale,
            sampling_ratio=2
        )
        return lines_feat


class ImageRegionExtractor(nn.Module):
    def __init__(self, scale, output_size):
        super().__init__()
        self.scale = scale
        self.output_size = output_size
    
    def forward(self, images, line_bboxes):
        rois = tensor_convert_to_roi_format(line_bboxes)
        images_feat = roi_align(
            input=images,
            boxes=rois,
            output_size=self.output_size,
            spatial_scale=self.scale,
            sampling_ratio=1
        )
        images_feat = images_feat.reshape(images_feat.shape[0], -1)
        view_shape = line_bboxes.shape[:2]
        images_feat = images_feat.view(*view_shape, -1)
        return images_feat


