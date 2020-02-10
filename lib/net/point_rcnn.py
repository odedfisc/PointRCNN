from web_visualizer import pc_show
import numpy as np
import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet
from lib.config import cfg


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                if cfg.RCNN.USE_RPN_FEATURES:
                    self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels,
                                            use_xyz=use_xyz)
                elif cfg.RCNN.USE_SURFACE_FEATURES:
                    self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=8,
                                            use_xyz=use_xyz)

                else:
                    self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=0,
                                            use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            else:
                raise NotImplementedError

    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)

            # rcnn inference
            if cfg.RCNN.ENABLED:
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']
                    clusters_mask = (input_data['pts_clusters'] > 0).float()
                    rpn_scores_raw = rpn_cls[:, :, 0]
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float().unsqueeze(dim=2)
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                    # proposal layer
                    if cfg.RCNN.USE_CLUSTERING_MASK:
                        rois, roi_scores_raw = self.rpn.proposal_layer(clusters_mask, rpn_reg, backbone_xyz)  # (B, M, 7)
                        seg_mask = clusters_mask
                    else:
                        rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)),
                                   'seg_mask': seg_mask,  # switch segmentation mask with clustering
                                   'roi_boxes3d': rois,
                                   'pts_surface_features': input_data['pts_surface_features'],
                                   'pts_depth': pts_depth}
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError
        # pc_show(np.concatenate((input_data['pts_input'][0].data.cpu().numpy(),
        #                         (input_data['pts_clusters'][0] != 0).astype(np.uint8) * 255), axis=1))
        return output
