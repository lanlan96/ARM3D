import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from nn_distance import nn_distance, huber_loss
import numpy as np
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

        # self.gu_conv = torch.nn.Conv2d(256, 256, (1, 1))

        # self.gu_bn = torch.nn.BatchNorm2d(256)

        # self.gu_conv2 = torch.nn.Conv2d(256, 128, (1, 1))
        # self.gu_bn2 = torch.nn.BatchNorm2d(128)

        # self.adamaxpool = nn.AdaptiveMaxPool2d((256,1))
        # self.gmax = torch.nn.Maxpool2d(kernel_size=2,stride=1)
        # self.att = Attention(128, 128)
        self.sa1 = SA_Layer(128, self.relation_pair)


        self.rn_conv1 = torch.nn.Conv1d(128, 128, 1)
        # self.rn_bn = torch.nn.BatchNorm1d(128)
        # self.rn_conv11 = torch.nn.Conv1d(128, 128, 1)
        # self.rn_conv1 = torch.nn.Conv1d(128, 128, 1)
        # self.rn_conv2 = torch.nn.Conv1d(128, 128, 1)
        self.rn_conv2 = torch.nn.Conv1d(128, 128, 1)


        self.rn_bn1 = torch.nn.BatchNorm1d(128)
        self.rn_bn2 = torch.nn.BatchNorm1d(128)
        self.rn_conv3 = torch.nn.Conv1d(128, 128, 1)
        self.rn_conv4 = torch.nn.Conv1d(128, 2, 1)
        
        self.rn_bn1_1 = torch.nn.BatchNorm1d(128)
        self.rn_bn2_1 = torch.nn.BatchNorm1d(128)
        self.rn_conv3_1 = torch.nn.Conv1d(128, 128, 1)
        self.rn_conv4_1 = torch.nn.Conv1d(128, 2, 1)

        #same instance
        # self.rn_bn1_2 = torch.nn.BatchNorm1d(128)
        # self.rn_bn2_2 = torch.nn.BatchNorm1d(128)
        # self.rn_conv3_2 = torch.nn.Conv1d(128, 128, 1)
        # self.rn_conv4_2 = torch.nn.Conv1d(128, 2, 1)

        #NOT USED
        #self.rn_bn1 = torch.nn.BatchNorm1d(256)
        #self.rn_bn2 = torch.nn.BatchNorm1d(128)
        #self.rn_conv3 = torch.nn.Conv1d(256, 128, 1)
        #self.rn_conv4 = torch.nn.Conv1d(128, 2, 1)

        #self.rn_bn1_1 = torch.nn.BatchNorm1d(256)
        #self.rn_bn2_1 = torch.nn.BatchNorm1d(128)
        #self.rn_conv3_1 = torch.nn.Conv1d(256, 128, 1)
        #self.rn_conv4_1 = torch.nn.Conv1d(128, 2, 1)



    def forward(self, feature, end_points):
        # prepare features for relation pairs
        feature_ori = feature
        bs, fea_in, num_proposal = feature.shape  # [B, feat_in, proposal_num]
        feature = feature.permute(0, 2, 1)  # [B, proposal_num, feat_in]  [8, 256, 128] 128确定吗
        """
        NEW
        """
        idx_obj = end_points['idx_obj'] #(8,256)
        sum_1 = end_points['sum_1'] #(8)
        print("SUM 1:",sum_1)
        idx_obj_3 = idx_obj.unsqueeze(1).repeat(1,num_proposal, 1) #(8,256,256)
        idx_all = torch.zeros((1,num_proposal,self.relation_pair)).cuda().long()
        for i, num in enumerate(sum_1):
            if num<self.relation_pair:
                num=num_proposal
            obj_idx = torch.randint(0, num, (1,num_proposal, self.relation_pair)).cuda().long() #(1,256, 8)
            idx_all = torch.cat((idx_all, obj_idx), dim=0)
        idx_all = idx_all[1:,:,:] #(8,256,8)
        _idx_j_batch = batched_index_select(idx_obj_3, idx_all, dim=2)
        # print("IDX SHAPE:",_idx_j_batch)

        # Use center point for distance computation
        # pred_center = end_points['center']
        pred_center = end_points['aggregated_vote_xyz'] #(batch_size, num_proposal, 3)
        _pred_center_i = pred_center.view(bs, 1, num_proposal, 3).repeat(1, num_proposal, 1, 1)
        _pred_center_j = pred_center.view(bs, num_proposal, 1, 3).repeat(1, 1, num_proposal, 1)

        # dist = (_pred_center_j[:, :, :, :3] - _pred_center_i[:, :, :, :3]).pow(2)
        
        # dist = torch.sqrt(torch.sum(dist, -1)) * (-1)  #TODO: 为什么要*-1啊->目的就是为了取topk个的时候可以直接取到最近的k个
        # dist = torch.sqrt(dist)
        # print("max_dist:",torch.max(dist))
        # print("min_dist:",torch.min(dist))
        # print(dist)
        # get index j for i
        self.max_paris = self.relation_pair
        # if self.random:
        #     if self.max_paris == num_proposal: #256
        #         print("self.max_paris == num_proposal")
        #         _idx_j_batch = torch.arange(0,num_proposal,dtype=torch.int) #(0~255)
        #         _idx_j_batch = _idx_j_batch.expand(bs, num_proposal, self.max_paris).cuda().long()
        #     else:
        # _idx_j_batch = torch.randint(0, num_proposal, (bs, num_proposal, self.max_paris), dtype=torch.int).cuda().long()
        # else:
        #     # select the nearest proposals as relation pairs
        #     _, _idx_j = torch.topk(dist, k=self.relation_pair + 1) #TODO: 从最后一维度去取前k个的吗
        #     _idx_j_batch = _idx_j[:, :, 1:] #(bs,num_proposal,k)
        _idx_j_batch_reshape = _idx_j_batch.reshape((bs * num_proposal * self.max_paris))

        # get pair of features
        _feature_i = feature.unsqueeze(2).repeat((1, 1, self.max_paris, 1))  # [B, proposal_num, self.max_pairs, feat_in]
        _range_for_bs = torch.arange(bs).unsqueeze(1).repeat((1, num_proposal * self.max_paris)).reshape((bs * num_proposal * self.max_paris))
        _feature_j = feature[_range_for_bs, _idx_j_batch_reshape, :].reshape((bs, num_proposal, self.max_paris, fea_in))
        # relation_u = _feature_i.add(_feature_j)  # [B, proposal_num, max_pairs, feat_in]  ori
        # if self.use_edgeconv:
        relation_tmp = _feature_i.sub(_feature_j)
        relation_u = torch.cat((_feature_i,relation_tmp),-1)

        #TODO: 论文里我们用的不是concat吗
        # print("idxes.shape: ", idxes.shape)
        end_points['nearest_n_index'] = _idx_j_batch

        relation_local = relation_u.permute(0,3,1,2) # [B, feat_in, proposal_num, max_pairs]
        relation_local = F.relu(self.local_bn1(self.local_conv1(relation_local)))
        relation_local = F.relu(self.local_bn2(self.local_conv2(relation_local)))
        # relation_local = F.adaptive_max_pool2d(relation_local, (num_proposal,1))
        # relation_local = relation_local.squeeze(3) #[B, feat_in, proposal_num]
        assert(len(relation_local.size())==4)


        """
            g_theta
        """
        # get relation feature for each object

        # relation_u = relation_u.permute(0, 3, 1, 2)  # [B, feat_dim, proposal_num, pairs_num] [8, 128, 256, 3]

        # print("relation_u shape:", relation_u.shape)
        # gu_output = self.gu_conv(relation_u)  # [B, feat_dim, proposal_num, pairs_num][8, 256, 256, 3]
        # gu_output = (self.gu_bn(gu_output))  # debug

        # gu_output = self.gu_conv(relation_u)  # [B, feat_dim, proposal_num, pairs_num][8, 256, 256, 3]
        # gu_output = F.relu(self.gu_bn(gu_output))  # debug
        # gu_output = self.gu_conv2(gu_output)  # [B, feat_dim, proposal_num, pairs_num][8, 256, 256, 3]
        # gu_output_copy = gu_output  # [B, feat_dim, proposal_num, pairs_num]

        #TODO: 加不加relu bn什么的？
        # gu_output = F.relu(self.gu_bn2(gu_output))  # debug
        # gu_output = self.adamaxpool(gu_output).squeeze(-1) #[B, feat_dim, proposal_num]

        # gu_output = self.gmax(gu_output)


        #TODO: 不经过relu吗

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
        print("ATT MAX:",torch.max(att_w))
        print("ATT MIN:",torch.min(att_w))
        print("ATT SUM:",torch.sum(att_w))
        print("ATT MEAN:",torch.mean(att_w))
        print("ATT VAR:",torch.var(att_w))

        att_w = att_w.reshape(bs, num_proposal, self.max_paris)
        att_w_sum = torch.sum(att_w , dim=2) #(bs, num_proposal)
        att_w_sum = att_w_sum.unsqueeze(2).expand(-1,-1,self.max_paris) #(bs, num_proposal, self.max_paris)
        att_w = att_w.div(att_w_sum) #(bs, num_proposal, self.max_paris)

        # top_w,_ = torch.topk(att_w, k=10, dim=2)
        # print(top_w[0][0])
        #visualize attention
        # plt.figure(figsize=(40, 40))
        # for i in range(8):  # 可视化了32通道
        #     ax = plt.subplot(1, 8, i+1)
        #     # ax.set_title('Feature {}'.format(i))
        #     ax.axis('off')
        #     # ax.set_title('attention-image')

        #     plt.imshow(att_w.data.cpu().numpy()[i,:,:],cmap='jet')
        #    if i ==7:
        #       save_str = "attention_vis/{}_W_jet.png".format(str(i))
        #       plt.savefig(save_str)
        # plt.show()  # 图像每次都不一样，是因为模型每次都需要前向传播一次，不是加载的与训练模型



        # num_w = att_w.shape[0]*att_w.shape[1]*att_w.shape[2]
        # print("att_w shape:",att_w.shape)
        # # b = torch.randn((3,3))
        # t1 =att_w >0.8
        # t1_2 = att_w <0.8
        # t2 =att_w >0.6
        # t2_3 =att_w<0.6
        # t3 =att_w>0.4
        # t3_4 = att_w<0.4
        # t4= att_w>0.2
        # t4_5 = att_w<0.2
        # t5=att_w>0.1
        # t5_6 =att_w<0.15

        # t08=t1
        # t0608=t1_2 & t2
        # t0406 = t3 & t2_3
        # t0204 = t3_4 & t4
        # t0102 = t4_5 & t5
        # tl01 = t5_6

        # # b2 = a>1
        # v08 = torch.nonzero(t08)
        # v0608 = torch.nonzero(t0608)
        # v0406 = torch.nonzero(t0406)
        # v0204 = torch.nonzero(t0204)
        # v0102 = torch.nonzero(t0102)
        # vl01 = torch.nonzero(tl01)


        # print("0.8-1.0:",v08.shape[0],v08.shape[0]/num_w)
        # print("0.6-0.8:",v0608.shape[0],v0608.shape[0]/num_w)
        # print("0.4-0.6:",v0406.shape[0],v0406.shape[0]/num_w)
        # print("0.2-0.4:",v0204.shape[0],v0204.shape[0]/num_w)
        # print("0.1-0.2:",v0102.shape[0],v0102.shape[0]/num_w)
        # print("0.0-0.15:",vl01.shape[0],vl01.shape[0]/num_w)

        # #砍掉所有0.15以下的
        # att_w = att_w.cpu()
        # att_w[np.where(att_w<0.07)]=0
        # att_w= att_w.cuda()

        att_w = att_w.unsqueeze(-1).repeat(1, 1, 1, gu_output.shape[1]) #(bs, num_proposal, self.max_paris, feature)


        # print("att_max:", torch.max(att_w)," att_min:", torch.min(att_w), 
        #         "att_avg:",torch.mean(att_w))
        # # print("att_w:",att_w)
        gu_output = gu_output.permute(0,2,3,1) #(bs, num_proposal, self.max_paris, feature)
        print("att_w:",att_w.shape)
        print("gu_output:",gu_output.shape)
        # print("MAX gu_output:",torch.max(gu_output))
        # print("MIN gu_output:",torch.min(gu_output))
        # print("SUM gu_output:",torch.sum(gu_output)) 
        gu_output = att_w * gu_output #
        # print("MAX gu_output2:",torch.max(gu_output))
        # print("MIN gu_output2:",torch.min(gu_output))
        # print("SUM gu_output2:",torch.sum(gu_output)) 
        # unsqueeze_h = torch.cat((hidden_state, feature_ori),1)
        unsqueeze_h =gu_output
        # unsqueeze_h = F.relu(self.rn_bn(self.rn_conv1(unsqueeze_h)))  # [bs, fea_channel, proposal_num] [B, 128, 256]
        # output = self.rn_conv11(unsqueeze_h)

        unsqueeze_h = torch.sum(gu_output,dim=2).squeeze(2).permute(0,2,1)   # [B, feat_dim, proposal_num]                                                                                                                                                                                                                                                                                                                                                                                                    
        #暂时不需要mean,因为做了maxpool
        # unsqueeze_h = torch.mean(gu_output, 3)  # [B, feat_dim, proposal_num] [8, 256, 256]
        # unsqueeze_h = gu_output

        output = self.rn_conv1(unsqueeze_h)  # [bs, fea_channel, proposal_num] [B, 128, 256]
        #TODO: 就经过一个rn_conv1啊，太简单了吧
        
        # output = torch.sigmoid(torch.log(torch.abs(output)))
        # output = F.sigmoid(output)
        # print("MAX output:",torch.max(output))
        # print("MIN output:",torch.min(output))
        # print("SUM output:",torch.sum(output))  

        end_points['rn_feature'] = output

        """
            predict relation labels
        """
        # get relationships for all pairs
        # _, u_inchannel, _, _ = relation_u.shape
        # relation_all = relation_u.view(bs, u_inchannel, num_proposal * self.max_paris)
        #TODO: 这里不应该是拿gu_output做预测吗
        # [B, feat_dim, proposal_num, pairs_num]
        _, u_inchannel, _, _ = relation_local.shape
        relation_all = relation_local.view(bs, u_inchannel, num_proposal * self.max_paris)
        relation_all = self.rn_conv2(relation_all) #self.rn_conv2 = torch.nn.Conv1d(128, 128, 1)

        #_, u_inchannel, _, _ = gu_output.shape
        #relation_all = gu_output.view(bs, u_inchannel, num_proposal * self.max_paris)

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
        #师姐原先是不要same_instance的
        if 'same_instance' in self.relation_type:
            relation_all_2 = F.relu(self.rn_bn1_2(relation_all))
            relation_all_2 = self.rn_conv3_2(relation_all_2)
            relation_all_2 = F.relu(self.rn_bn2_2(relation_all_2))
            logits_2 = self.rn_conv4_2(relation_all_2)  # same instance
            end_points['rn_logits_2'] = logits_2

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
        self.softmax = nn.Softmax(dim=-1)  #TODO: 到底要不要指定dim=-1
        # self.softmax = nn.Softmax()  #TODO: 到底要不要指定dim=-1

        # self.softmax = nn.Sigmoid()

        self.relation_pair = relation_pair
        #
        self.tanh = nn.Tanh()

    def forward(self, x, x_key): # (B, C , N)进来
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

        print("ENERGY:",energy)
        print("ENERGYmin:",torch.min(energy))
        print("ENERGYmax:",torch.max(energy))
        print("ENERGYmean:",torch.mean(energy))
        # print("ENERGYvar:",torch.var(energy))

        attention = self.softmax(energy)
        # print("softmax:",attention)
        # print("softmax min:",torch.min(attention))
        # print("softmax max:",torch.max(attention))
        # print("softmax mean:",torch.mean(attention))
        # attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True)) 
        attention_r =attention.clone()
        #([8, 256, 256])

        # bs*num_proposal, pairs, pairs

        # for i in range(attention.shape[0]):
        #     print("att_trans:",attention[i,:,0])
        #     print("MAX:",torch.max(attention[i,:,0]))
        #     print("MIN:",torch.min(attention[i,:,0]))
        #     print("SUM:",torch.sum(attention[i,:,0]))
        #     print("\n")
        #get topk
        # _, _idx_k = torch.topk(attention, k=self.relation_pair+1, dim = 1)

        
        # # # 特征输出可视化
        # _idx_j_batch_reshape = _idx_k.reshape((8 * 256 * self.relation_pair))

        # _range_for_bs = torch.arange(8).unsqueeze(1).repeat((1, 256 * self.relation_pair)).reshape((8 * 256 * self.relation_pair))
        # _range_for_p = torch.arange(256).unsqueeze(0).repeat((8, 1)).unsqueeze(-1).repeat((1, 1, self.relation_pair)).reshape((8 * 256 * self.relation_pair))
        # attention[_range_for_bs, _range_for_p, _idx_j_batch_reshape] = 10.0

        # # 展示transformer选择的效果
        # # copy一下attention给nearest
        # n_idx_j_batch = n_idx_j_batch.reshape((8 * 256 * self.relation_pair))

        # attention_n = attention.clone()
        # attention_n[:,:,:] = 0.0
        # attention_n[_range_for_bs, _range_for_p, n_idx_j_batch] =10.0

        
        # print(attention)
        # print("attention shape:", attention.shape)
        # attention_np = attention.cpu().detach().numpy()
        # _idx_k_np = _idx_k.cpu().detach().numpy()
        # print(_idx_k_np.shape)
        # print(attention_np.shape)
        # attention_np [_idx_k_np] = 1.0
    
        # plt.figure(figsize=(25, 5))
        # for i in range(16):  # 可视化了32通道
        #     ax = plt.subplot(2, 8, i+1)
        #     # ax.set_title('Feature {}'.format(i))
        #     ax.axis('off')
        #     # ax.set_title('attention-image')
        #     if i<8:
        #         plt.imshow(attention.data.cpu().numpy()[i,:,:],cmap='jet')
        #     else:
        #         plt.imshow(attention_n.data.cpu().numpy()[i-8,:,:],cmap='jet')
        #     if i ==15:
        #         save_str = "attention_vis/{}_transformer_pointer_t_n_test.png".format(str(i))
        #         plt.savefig(save_str)
        # plt.show()  # 图像每次都不一样，是因为模型每次都需要前向传播一次，不是加载的与训练模型

        # plt.figure(figsize=(50, 50))
        # for i in range(8):  # 可视化了32通道
        #     ax = plt.subplot(1, 8, i+1)
        #     # ax.set_title('Feature {}'.format(i))
        #     ax.axis('off')
        #     # ax.set_title('attention-image')

        #     plt.imshow(attention.data.cpu().numpy()[i*256,:,:1],cmap='jet')
        #     if i ==7:
        #         save_str = "attention_vis/{}_transformer_ori_nearest_differentbs2.png".format(str(i))
        #         plt.savefig(save_str)
        # plt.show()  # 图像每次都不一样，是因为模型每次都需要前向传播一次，不是加载的与训练模型

        # sys.exit()

        # b, c, n
        #b,c,n * b,n,1 -> b, c, 1
        x_r = torch.bmm(x_v, attention_r.unsqueeze(-1))
        # print("x_r:",x_r.shape)
        # print("attention:",attention.shape)
        # x_r = x_r.squeeze(-1) #b, c
        # x_r = x_r.reshape(bs)
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