# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils



'''get objectness index'''
from nn_distance import nn_distance
NEAR_THRESHOLD = 0.3




class ObjectnessModule_Pred(nn.Module):
    def __init__(self, feat_dim):
        super().__init__() 

        self.conv1 = torch.nn.Conv1d(feat_dim,64,1)
        self.conv2 = torch.nn.Conv1d(64,32,1)
        self.conv3 = torch.nn.Conv1d(32,2,1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(32)

    def forward(self, features, end_points):
 
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)
        net_transposed = net.transpose(2,1) # (batch_size, num_proposal, 2 )
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:,:,0:2]
        end_points['objectness_scores'] = objectness_scores
        
        # print("objectness_scores:",objectness_scores)
        obj_pred_val = torch.argmax(objectness_scores, 2) # B,K  

        return obj_pred_val, end_points




if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
