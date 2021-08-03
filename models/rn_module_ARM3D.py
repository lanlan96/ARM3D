import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from nn_distance import nn_distance, huber_loss
import numpy as np
from SE import SE
from CBAM import CBAM
import matplotlib.pyplot as plt


class RNModule(nn.Module):
    def __init__(self, relation_pair=3, relation_type='same_category', random=False):
        super().__init__()
        self.relation_pair = relation_pair
        self.relation_type = relation_type
        self.random = random
        print('relation pair: {}'.format(self.relation_pair))
        print("FLAGS.relation_type: ", self.relation_type)
        print("random: ", self.random)
        # self.gu_conv = torch.nn.Conv2d(128, 256, (1, 1))
        self.local_conv1= torch.nn.Conv2d(256, 256 , (1, 1)) 
        self.local_bn1 = torch.nn.BatchNorm2d(256)

        self.local_conv2= torch.nn.Conv2d(256, 128 , (1, 1)) 
        self.local_bn2 = torch.nn.BatchNorm2d(128)

        self.sa1 = SA_Layer(128, self.relation_pair)


        self.rn_conv1 = torch.nn.Conv1d(128, 128, 1)

        self.rn_conv2 = torch.nn.Conv1d(128, 128, 1)


        self.rn_bn1 = torch.nn.BatchNorm1d(128)
        self.rn_bn2 = torch.nn.BatchNorm1d(128)
        self.rn_conv3 = torch.nn.Conv1d(128, 128, 1)
        self.rn_conv4 = torch.nn.Conv1d(128, 2, 1)
        
        self.rn_bn1_1 = torch.nn.BatchNorm1d(128)
        self.rn_bn2_1 = torch.nn.BatchNorm1d(128)
        self.rn_conv3_1 = torch.nn.Conv1d(128, 128, 1)
        self.rn_conv4_1 = torch.nn.Conv1d(128, 2, 1)





    def forward(self, feature, end_points):
        # prepare features for relation pairs
        feature_ori = feature
        bs, fea_in, num_proposal = feature.shape  # [B, feat_in, proposal_num]
        feature = feature.permute(0, 2, 1)  # [B, proposal_num, feat_in]  

        idx_obj = end_points['idx_obj'] #(8,256)
        sum_1 = end_points['sum_1'] #(8)

        idx_obj_3 = idx_obj.unsqueeze(1).repeat(1,num_proposal, 1) #(8,256,256)
        idx_all = torch.zeros((1,num_proposal,self.relation_pair)).cuda().long()
        for i, num in enumerate(sum_1):
            if num<self.relation_pair:
                num=num_proposal
            obj_idx = torch.randint(0, num, (1,num_proposal, self.relation_pair)).cuda().long() #(1,256, 8)
            idx_all = torch.cat((idx_all, obj_idx), dim=0)
        idx_all = idx_all[1:,:,:] #(8,256,8)
        _idx_j_batch = batched_index_select(idx_obj_3, idx_all, dim=2)

        pred_center = end_points['aggregated_vote_xyz'] #(batch_size, num_proposal, 3)
        _pred_center_i = pred_center.view(bs, 1, num_proposal, 3).repeat(1, num_proposal, 1, 1)
        _pred_center_j = pred_center.view(bs, num_proposal, 1, 3).repeat(1, 1, num_proposal, 1)



        self.max_paris = self.relation_pair

        _idx_j_batch_reshape = _idx_j_batch.reshape((bs * num_proposal * self.max_paris))

        # get pair of features
        _feature_i = feature.unsqueeze(2).repeat((1, 1, self.max_paris, 1))  # [B, proposal_num, self.max_pairs, feat_in]
        _range_for_bs = torch.arange(bs).unsqueeze(1).repeat((1, num_proposal * self.max_paris)).reshape((bs * num_proposal * self.max_paris))
        _feature_j = feature[_range_for_bs, _idx_j_batch_reshape, :].reshape((bs, num_proposal, self.max_paris, fea_in))

        relation_tmp = _feature_i.sub(_feature_j)
        relation_u = torch.cat((_feature_i,relation_tmp),-1)


        end_points['nearest_n_index'] = _idx_j_batch

        relation_local = relation_u.permute(0,3,1,2) # [B, feat_in, proposal_num, max_pairs]
        relation_local = F.relu(self.local_bn1(self.local_conv1(relation_local)))
        relation_local = F.relu(self.local_bn2(self.local_conv2(relation_local)))

        assert(len(relation_local.size())==4)




        """
            f_phi
        """
        gu_output = relation_local
        print("before mean:",gu_output.shape)
        
        attention_j = gu_output.permute(0,2,3,1)

        relation_key = feature.reshape((bs*num_proposal, fea_in)).unsqueeze(-1)

        relation_knn = attention_j.reshape((bs*num_proposal, self.max_paris, fea_in))
        relation_knn = relation_knn.permute(0,2,1) # b*n, feature, pairs

        global_relation, att_w = self.sa1(relation_knn, relation_key) 

        print("att_trans:",att_w)


        att_w = att_w.reshape(bs, num_proposal, self.max_paris)
        att_w_sum = torch.sum(att_w , dim=2) #(bs, num_proposal)
        att_w_sum = att_w_sum.unsqueeze(2).expand(-1,-1,self.max_paris) #(bs, num_proposal, self.max_paris)
        att_w = att_w.div(att_w_sum) #(bs, num_proposal, self.max_paris)





        att_w = att_w.unsqueeze(-1).repeat(1, 1, 1, gu_output.shape[1]) #(bs, num_proposal, self.max_paris, feature)


        gu_output = gu_output.permute(0,2,3,1) #(bs, num_proposal, self.max_paris, feature)

        gu_output = att_w * gu_output #
 

        unsqueeze_h =gu_output


        unsqueeze_h = torch.sum(gu_output,dim=2).squeeze(2).permute(0,2,1)   # [B, feat_dim, proposal_num]                                                                                                                                   

        output = self.rn_conv1(unsqueeze_h)  # [bs, fea_channel, proposal_num] [B, 128, 256]
 
        


        end_points['rn_feature'] = output

        """
            predict relation labels
        """

        _, u_inchannel, _, _ = relation_local.shape
        relation_all = relation_local.view(bs, u_inchannel, num_proposal * self.max_paris)
        relation_all = self.rn_conv2(relation_all) #self.rn_conv2 = torch.nn.Conv1d(128, 128, 1)


        if 'same_category' in self.relation_type:
            relation_all_0 = F.relu(self.rn_bn1(relation_all))
            relation_all_0 = self.rn_conv3(relation_all_0)
            relation_all_0 = F.relu(self.rn_bn2(relation_all_0))
            logits_0 = self.rn_conv4(relation_all_0)  # (bs, 2, num_proposal*max_pairs) same category
            end_points['rn_logits_0'] = logits_0 

        if 'support' in self.relation_type:
            relation_all_1 = F.relu(self.rn_bn1_1(relation_all))
            relation_all_1 = self.rn_conv3_1(relation_all_1)
            relation_all_1 = F.relu(self.rn_bn2_1(relation_all_1))
            logits_1 = self.rn_conv4_1(relation_all_1)  # support
            # probs = torch.nn.softmax(logits, 2)
            end_points['rn_logits_1'] = logits_1


        return end_points




class SA_Layer(nn.Module):
    def __init__(self, channels,  relation_pair=8):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1)
        # self.q_conv.weight = self.k_conv.weight
        # self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  


        self.relation_pair = relation_pair
        #
        self.tanh = nn.Tanh()

    def forward(self, x, x_key): # (B, C , N)
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, 1
        x_k = self.k_conv(x_key)
        # print(self.q_conv.weight)
        x_v = self.v_conv(x) #b,c,n
        # b, n, 1
        # energy = torch.bmm(x_q, x_k) # b, n, 1
        energy = torch.bmm(self.tanh(x_q), self.tanh(x_k)) # b, n, 1


        energy = energy.squeeze(-1) # b, n
 

        attention = self.softmax(energy)

        attention_r =attention.clone()



        x_r = torch.bmm(x_v, attention_r.unsqueeze(-1))

        x_r = self.act(self.after_norm(self.trans_conv(x_key - x_r))) #([2048, 128, 1])
        
        x_final = x_key + x_r
        x_final = x_final.squeeze(-1)
        return x_final, attention


def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)