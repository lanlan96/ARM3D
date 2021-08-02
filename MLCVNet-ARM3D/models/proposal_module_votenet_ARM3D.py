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
from obj_mlp import ObjectnessModule
from obj_mlp import ObjectnessModule_Pred


from rn_module_edgeconv_transformer_obj import RNModule as RNOBJ
# from rn_module_edgeconv_transformer_obj_noatt import RNModule as RNOBJ
# from rn_module_edgeconv_transformer_obj_global import RNModule as RNOBJ


# from rn_module_ori_transformer_10J_mean_subcat_pairs_global import RNModule
# from rn_module_ori_transformer_10J_mean_all_nosoft import RNModule
# from rn_module_ori_transformer_simple import RNModule
from rn_module_edgeconv_transformer import RNModule

# from rn_module_transformer_fenduan import RNModule
# from rn_module_ori import RNModule
# from rn_module_ori_transformer_10J_mean import RNModule  #
# from rn_module_pointer_knearest_gupred import RNModule  #
# from rn_module_attention_add import RNModule
# from rn_module_ori_transformer_nearest_j_knearest import RNModule
# from rn_module_ori_transformer_sigmoid_mean import RNModule  #
# from rn_module_ori import RNModule  #
# from rn_module_ori_transformer_knearest import RNModule  #
# from rn_module_pointer_j import RNModule  #
# from rn_module_edgeconv_wATT_gupred_v4_trans import RNModule  #
# from rn_module_edgeconv_wATT_gupred import RNModule

# from rn_module_attention_beforegu import RNModule
# from rn_module_attzention_gupred import RNModule
# from rn_module_ori_transformer_knearest import RNModule

from SE import SE
from CBAM import CBAM


'''获取objectness index'''
from nn_distance import nn_distance
NEAR_THRESHOLD = 0.3

def get_index(objectness_label):
    a = torch.arange(objectness_label.shape[1]).unsqueeze(0).repeat((objectness_label.shape[0],1))

    obj_all_ind = a[objectness_label==1] #所有的index在一个tensor里
    # print(obj_all_ind)
    sum_0 = torch.sum(objectness_label, dim=1).cuda().long()
    # print("COUNT:",sum_0)
    start =0
    _idx_batch = torch.zeros((1, 256)).cuda().long()
    for i, num in enumerate(sum_0):
        # print(i,num)
        tmp = obj_all_ind[start:start+num].cuda().long()
        # tensor256 = torch.zeros(256).cuda().long()
        tensor256 = torch.arange(0, 256, dtype=torch.int).cuda().long()
        # tensor256 +=256 
        tensor256[:num] = tmp
        tensor256 = tensor256.unsqueeze(0)  #(1,256)
        # print("tensor256:",tensor256.shape)
        # print("tensor256:",tensor256)
        _idx_batch = torch.cat((_idx_batch,tensor256),dim=0)
        # print("_idx_batch:",_idx_batch)
        start +=num
        # print( start)
    _idx_batch=_idx_batch[1:,:]
    return _idx_batch, sum_0


def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:,:,5:5+num_heading_bin]  #num_heading_bin (Scannet:1)
    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10  Scannet:18(num_class)
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points

def decode_scores_pred(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    # objectness_scores = net_transposed[:,:,0:2]
    # end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,0:3] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:,:,3:3+num_heading_bin]  #num_heading_bin (Scannet:1)
    heading_residuals_normalized = net_transposed[:,:,3+num_heading_bin:3+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,3+num_heading_bin*2:3+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = net_transposed[:,:,3+num_heading_bin*2+num_size_cluster:3+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,3+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10  Scannet:18(num_class)
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, relation_pair=3, relation_type=['same_category'], random=False, is_se=False,use_cbam=False, use_global=False, use_cluster=False):
        super().__init__() 
        self.is_se = is_se
        self.use_cbam =use_cbam
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal #128
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.use_cluster = use_cluster

        self.use_global = use_global
        if self.is_se:
            self.se = SE(256, 16)  #f_phi's output channel
        
        if self.use_cbam:
            self.cbam = CBAM( 256 )

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        if self.use_cluster:
            # Vote clustering 参数1
            # self.vote_aggregation2 = PointnetSAModuleVotes( 
            #         npoint=relation_pair,
            #         radius=0.5,
            #         nsample=32,
            #         mlp=[self.seed_feat_dim, 128, 128, 128],
            #         use_xyz=True,
            #         normalize_xyz=True
            #     )
            #参数2
            self.vote_aggregation2 = PointnetSAModuleVotes( 
                    npoint=relation_pair,
                    radius=0.3,
                    nsample=16,
                    mlp=[self.seed_feat_dim, 128, 128, 128],
                    use_xyz=True,
                    normalize_xyz=True
                )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        #self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv1 = torch.nn.Conv1d(128*2, 128, 1)
        if self.use_global:
            self.conv1g = torch.nn.Conv1d(128*3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.rnet = RNModule(relation_pair=relation_pair, relation_type=relation_type, random=random)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3) vote_xyz
            features: (B,C,K) vote_feature
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            if self.use_cluster:
                xyz_cluster, features_cluster, fps_inds_cluster = self.vote_aggregation2(xyz, features)
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        if self.use_cluster:
            end_points['cluster_vote_xyz'] =xyz_cluster
            end_points['cluster_vote_feature'] =features_cluster
            end_points['cluster_vote_inds'] =fps_inds_cluster

        # --------- PROPOSAL GENERATION ---------

        # Add rn feature
        # print("FEATURES:",features.grad_fn.next_functions[0][0])
        # print(out.grad_fn.next_functions[0][0])
        end_points = self.rnet(features, end_points)
        rn_feature = end_points['rn_feature']
        
        print("net: {}, {}".format(torch.min(features), torch.max(features)))
        print("rn: {}, {}".format(torch.min(rn_feature), torch.max(rn_feature)))
        # print("net_features:",features.shape) #torch.Size([8, 128, 256])
        # print("rn_features:",rn_feature.shape) #torch.Size([8, 128, 256])
        
        if self.use_global:
            global_feature =end_points['global_feature']
            features = torch.cat((features, rn_feature,global_feature),dim=1)
        else:
            features = torch.cat((features, rn_feature), 1) #256维度吗?进入conv1之前 torch.Size([8, 256, 256])

        #加入SE BOLCK
        if self.is_se:
            features = torch.unsqueeze(features,-1)
            # print("se input:",output.shape) #([8, 256, 256])
            coefficient = self.se(features)
            features *= coefficient
            features = torch.squeeze(features,-1)
        #CBAM
        if self.use_cbam :
            features = torch.unsqueeze(features,-1)
            # print("output shape:",output.shape)
            features = self.cbam(features)
            # print("output shape:",output.shape)
            features = torch.squeeze(features,-1)
            # print("output shape:",output.shape)

        # print("concat_features:",features.shape) #torch.Size([8, 256, 256])
        # features = features + rn_feature
        if self.use_global:
            net = F.relu(self.bn1(self.conv1g(features))) 
        else:
            net = F.relu(self.bn1(self.conv1(features))) 
        
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)'
        # print("WEIGHT:",self.conv3.weight)
        #TODO: train时候取消
        print("cat(fetures,rn_feature): {}, {}".format(torch.min(net), torch.max(net)))

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return end_points




class ProposalModule_pred(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, relation_pair=3, relation_type=['same_category'], random=False):
        super().__init__() 
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal #128
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
   
        self.conv1 = torch.nn.Conv1d(128*2, 128, 1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.rnet = RNOBJ(relation_pair=relation_pair, relation_type=relation_type, random=random)

        self.onet = ObjectnessModule_Pred(128)


    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3) vote_xyz
            features: (B,C,K) vote_feature
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        #TODO:待注释
        gt_center = end_points['center_label'][:,:,0:3]
        dist1, ind1, dist2, _ = nn_distance(xyz, gt_center) # dist1: BxK, dist2: BxK2
        euclidean_dist1 = torch.sqrt(dist1+1e-6)
        B = gt_center.shape[0]
        K = xyz.shape[1]
        objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
        objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1  #(8, 256) 1

        ''' 加载Votenet预测objectness '''
       
        objectness_pred, end_points = self.onet(features, end_points)
        # np.set_printoptions(threshold=np.inf)
        print("\nGT:", torch.sum(objectness_label,dim =1))
        obj_acc = torch.sum(objectness_pred==objectness_label.long()).item()/(objectness_label.shape[0]*objectness_label.shape[1])
        print("OBJ ACCCCCCCCCCCCCCC:",obj_acc)


        idx_obj, sum_1 = get_index(objectness_pred)
        end_points['idx_obj'] = idx_obj
        end_points['sum_1'] = sum_1
        # --------- PROPOSAL GENERATION ---------
        # Add rn feature
        end_points = self.rnet(features, end_points)
        rn_feature = end_points['rn_feature']
        
        print("net: {}, {}".format(torch.min(features), torch.max(features)))
        print("rn: {}, {}".format(torch.min(rn_feature), torch.max(rn_feature)))

        features = torch.cat((features, rn_feature), 1) #256维度吗?进入conv1之前 torch.Size([8, 256, 256])

        # print("concat_features:",features.shape) #torch.Size([8, 256, 256])
        # features = features + rn_feature

        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)'
        # print("WEIGHT:",self.conv3.weight)
        print("cat(fetures,rn_feature): {}, {}".format(torch.min(net), torch.max(net)))

        end_points = decode_scores_pred(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return end_points



class ProposalModule_restore(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, relation_pair=3, relation_type=['same_category'], random=False):
        super().__init__() 
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal #128
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
   
        self.conv1 = torch.nn.Conv1d(128*2, 128, 1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.rnet = RNOBJ(relation_pair=relation_pair, relation_type=relation_type, random=random)

        self.onet = ObjectnessModule(128, self.num_heading_bin, self.num_size_cluster, self.num_class)


    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3) vote_xyz
            features: (B,C,K) vote_feature
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        #TODO:待注释
        gt_center = end_points['center_label'][:,:,0:3]
        dist1, ind1, dist2, _ = nn_distance(xyz, gt_center) # dist1: BxK, dist2: BxK2
        euclidean_dist1 = torch.sqrt(dist1+1e-6)
        B = gt_center.shape[0]
        K = xyz.shape[1]
        objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
        objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1  #(8, 256) 1
        ''' 加载Votenet预测objectness '''
        with torch.no_grad():
            objectness_pred, end_points = self.onet(features, end_points)

        obj_acc = torch.sum(objectness_pred==objectness_label.long()).item()/(objectness_label.shape[0]*objectness_label.shape[1])
        print("OBJ ACC:",obj_acc)


        idx_obj, sum_1 = get_index(objectness_pred)
        end_points['idx_obj'] = idx_obj
        end_points['sum_1'] = sum_1
        # --------- PROPOSAL GENERATION ---------
        # Add rn feature
        end_points = self.rnet(features, end_points)
        rn_feature = end_points['rn_feature']
        
        print("net: {}, {}".format(torch.min(features), torch.max(features)))
        print("rn: {}, {}".format(torch.min(rn_feature), torch.max(rn_feature)))

        features = torch.cat((features, rn_feature), 1) #256维度吗?进入conv1之前 torch.Size([8, 256, 256])

        # print("concat_features:",features.shape) #torch.Size([8, 256, 256])
        # features = features + rn_feature

        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)'
        # print("WEIGHT:",self.conv3.weight)
        print("cat(fetures,rn_feature): {}, {}".format(torch.min(net), torch.max(net)))

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return end_points


class ProposalModule_objectness(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, relation_pair=3, relation_type=['same_category'], random=False, is_se=False,use_cbam=False, use_global=False):
        super().__init__() 
        self.is_se = is_se
        self.use_cbam =use_cbam
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal #128
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        self.use_global = use_global
        if self.is_se:
            self.se = SE(256, 16)  #f_phi's output channel
        
        if self.use_cbam:
            self.cbam = CBAM( 256 )

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        #self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv1 = torch.nn.Conv1d(128*2, 128, 1)
        if self.use_global:
            self.conv1g = torch.nn.Conv1d(128*3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.rnet = RNOBJ(relation_pair=relation_pair, relation_type=relation_type, random=random)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3) vote_xyz
            features: (B,C,K) vote_feature
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        ''' 获得objectness index '''
        gt_center = end_points['center_label'][:,:,0:3]
        dist1, ind1, dist2, _ = nn_distance(xyz, gt_center) # dist1: BxK, dist2: BxK2
        euclidean_dist1 = torch.sqrt(dist1+1e-6)
        B = gt_center.shape[0]
        K = xyz.shape[1]
        objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
        objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1  #(8, 256) 1
        # objectness_label[euclidean_dist1>=NEAR_THRESHOLD] = 1  #(8, 256) 0
        # print("SHAPE:",objectness_label.shape)
        # print("objectness_label:",torch.nonzero(objectness_label).shape)
        idx_obj, sum_1 = get_index(objectness_label)
        end_points['idx_obj'] = idx_obj
        end_points['sum_1'] = sum_1
        # --------- PROPOSAL GENERATION ---------
        # Add rn feature
        end_points = self.rnet(features, end_points)
        rn_feature = end_points['rn_feature']
        
        print("net: {}, {}".format(torch.min(features), torch.max(features)))
        print("rn: {}, {}".format(torch.min(rn_feature), torch.max(rn_feature)))
        # print("net_features:",features.shape) #torch.Size([8, 128, 256])
        # print("rn_features:",rn_feature.shape) #torch.Size([8, 128, 256])
        
        if self.use_global:
            global_feature =end_points['global_feature']
            features = torch.cat((features, rn_feature,global_feature),dim=1)
        else:
            features = torch.cat((features, rn_feature), 1) #256维度吗?进入conv1之前 torch.Size([8, 256, 256])

        #加入SE BOLCK
        if self.is_se:
            features = torch.unsqueeze(features,-1)
            # print("se input:",output.shape) #([8, 256, 256])
            coefficient = self.se(features)
            features *= coefficient
            features = torch.squeeze(features,-1)
        #CBAM
        if self.use_cbam :
            features = torch.unsqueeze(features,-1)
            # print("output shape:",output.shape)
            features = self.cbam(features)
            # print("output shape:",output.shape)
            features = torch.squeeze(features,-1)
            # print("output shape:",output.shape)

        # print("concat_features:",features.shape) #torch.Size([8, 256, 256])
        # features = features + rn_feature
        if self.use_global:
            net = F.relu(self.bn1(self.conv1g(features))) 
        else:
            net = F.relu(self.bn1(self.conv1(features))) 
        
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)'
        # print("WEIGHT:",self.conv3.weight)
        #TODO: train时候取消
        print("cat(fetures,rn_feature): {}, {}".format(torch.min(net), torch.max(net)))

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return end_points





class ProposalModule_OBJ(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, relation_pair=3, relation_type=['same_category'], random=False):
        super().__init__() 
        self.is_se = is_se
        self.use_cbam =use_cbam
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal #128
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim


        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        #self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv1 = torch.nn.Conv1d(128*2, 128, 1)

        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.rnet = RN_MODULE_ORI(relation_pair=relation_pair, relation_type=relation_type, random=random)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3) vote_xyz
            features: (B,C,K) vote_feature
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------

        # Add rn feature
        end_points = self.rnet(features, end_points)
        rn_feature = end_points['rn_feature']
        
        print("net: {}, {}".format(torch.min(features), torch.max(features)))
        print("rn: {}, {}".format(torch.min(rn_feature), torch.max(rn_feature)))
        # print("net_features:",features.shape) #torch.Size([8, 128, 256])
        # print("rn_features:",rn_feature.shape) #torch.Size([8, 128, 256])
        
        if self.use_global:
            global_feature =end_points['global_feature']
            features = torch.cat((features, rn_feature,global_feature),dim=1)
        else:
            features = torch.cat((features, rn_feature), 1) #256维度吗?进入conv1之前 torch.Size([8, 256, 256])


        net = F.relu(self.bn1(self.conv1(features))) 
        
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)'
        # print("WEIGHT:",self.conv3.weight)
        #TODO: train时候取消
        # print("cat(fetures,rn_feature): {}, {}".format(torch.min(net), torch.max(net)))

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return end_points


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
