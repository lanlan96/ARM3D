# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using MLCVNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 40000]')
parser.add_argument('--scene_name', default='scene0609_02_vh_clean_2.ply', help='Scene name. [default: scene0609_02_vh_clean_2.ply]')
parser.add_argument('--model', default='mlcvnet', help='Model for visualization. [default: mlcvnet,votenet,mlcvnet_3DRM,votenet_3DRM, votenet_ARM-3D,mlcvnet_ARM-3D]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')

FLAGS = parser.parse_args()

FLAGS.relation_pair = FLAGS.checkpoint_path.split('rn')[1].split('_')[0]
print( "PAIR_NUM:",FLAGS.relation_pair)
if not FLAGS.relation_pair.startswith('adaptive'):
    FLAGS.relation_pair = int(FLAGS.relation_pair)
# FLAGS.relation_pair = 8
FLAGS.relation_type = []
if 'same_category' in FLAGS.checkpoint_path:
    FLAGS.relation_type.append('same_category')
if 'support' in FLAGS.checkpoint_path:
    FLAGS.relation_type.append('support')
if 'same_instance' in FLAGS.checkpoint_path:
    FLAGS.relation_type.append('same_instance')
if 'random' in FLAGS.checkpoint_path:
    FLAGS.random = True
else:
    FLAGS.random = False



import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply_scannet
from ap_helper import parse_predictions

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

if __name__=='__main__':
    
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import DC # dataset config
    checkpoint_path = FLAGS.checkpoint_path
    
    # Make sure your scannet ply file get transformed first, use rotate_val_scans.py to transform
    # Put any scannet transformed ply file in the demo_files folder and put its file name here
    # Then run demo.py
    pc_path = os.path.join(demo_dir, FLAGS.scene_name)

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_input_channel = 0
    if FLAGS.model =='mlcvnet':
        MODEL = importlib.import_module('mlcvnet') # import network module
        net = MODEL.MLCVNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
            sampling='seed_fps', num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr).to(device)
    if FLAGS.model =='votenet':
        MODEL = importlib.import_module('votenet') # import network module
        net = MODEL.VoteNet(num_class=DC.num_class,
               num_heading_bin=DC.num_heading_bin,
               num_size_cluster=DC.num_size_cluster,
               mean_size_arr=DC.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling,
               relation_pair=FLAGS.relation_pair,
               relation_type=FLAGS.relation_type,
               random=FLAGS.random)
    if FLAGS.model =='votenet_3DRM':    
        MODEL = importlib.import_module('3DRM') # import network module
        net = MODEL.VoteNet_3DRM(num_class=DC.num_class,
               num_heading_bin=DC.num_heading_bin,
               num_size_cluster=DC.num_size_cluster,
               mean_size_arr=DC.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling,
               relation_pair=FLAGS.relation_pair,
               relation_type=FLAGS.relation_type,
               random=FLAGS.random)
    if FLAGS.model =='mlcvnet_3DRM':    
        MODEL = importlib.import_module('3DRM') # import network module
        net = MODEL.MLCVNet_3DRM(num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                num_proposal=FLAGS.num_target,
                input_feature_dim=num_input_channel,
                vote_factor=FLAGS.vote_factor,
                sampling=FLAGS.cluster_sampling,
                relation_pair=FLAGS.relation_pair,
                relation_type=FLAGS.relation_type,
                random=FLAGS.random)
    if FLAGS.model =='votenet_ARM-3D':
        MODEL = importlib.import_module('ARM3D') # import network module
        net = MODEL.VoteNet_ARM3D(num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                num_proposal=FLAGS.num_target,
                input_feature_dim=num_input_channel,
                vote_factor=FLAGS.vote_factor,
                sampling=FLAGS.cluster_sampling,
                relation_pair=FLAGS.relation_pair,
                relation_type=FLAGS.relation_type,
                random=FLAGS.random)
    if FLAGS.model =='mlcvnet_ARM-3D':
        MODEL = importlib.import_module('ARM3D') # import network module
        net = MODEL.MLCVNet_ARM3D(num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                num_proposal=FLAGS.num_target,
                input_feature_dim=num_input_channel,
                vote_factor=FLAGS.vote_factor,
                sampling=FLAGS.cluster_sampling,
                relation_pair=FLAGS.relation_pair,
                relation_type=FLAGS.relation_type,
                random=FLAGS.random)
    print('Constructed model:',FLAGS.model)
    
    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud 
    net.eval() # set model to eval mode (for bn and dp)
    point_cloud = read_ply_scannet(pc_path)
    pc = preprocess_point_cloud(point_cloud)
    print('Loaded point cloud data: %s'%(pc_path))
   
    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))
    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))
  
    dump_dir = os.path.join(demo_dir, FLAGS.scene_name.split('.')[0])
    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    MODEL.dump_results(end_points, dump_dir, DC, True)
    print('Dumped detection results to folder %s'%(dump_dir))
