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
parser.add_argument('--scene_name', default='scene0609_02', help='Scene name. [default: scene0609_02_vh_clean_2.ply]')
parser.add_argument('--checkpoint_path', default='demo_files/pretrained_mlcvnet_on_scannet.tar', help='Scene name')
parser.add_argument('--pc_root', default='/home/lyq/Dataset/ScanNet/scannet/', help='pc root')
parser.add_argument('--model', default='votenet_ARM3D', help='Model for visualization')
parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')

FLAGS = parser.parse_args()
FLAGS.relation_pair = 8
FLAGS.relation_type = []
FLAGS.relation_type.append('same_category')
FLAGS.random = True 



import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply_scannet, export_aligned_mesh
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


def get_GTlabel(end_points, scan_name, data_path = 'scannet/scannet_train_detection_data'):
    """
    Returns a dict with following keys:
        point_clouds: (N,3+C)
        center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
        sem_cls_label: (MAX_NUM_OBJ,) semantic class index
        angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        angle_residual_label: (MAX_NUM_OBJ,)
        size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        size_residual_label: (MAX_NUM_OBJ,3)
        box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        point_votes: (N,3) with votes XYZ
        point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
        scan_idx: int scan index in scan_names list
        pcl_color: unused
    """
    MAX_NUM_OBJ = 64
    num_points = 40000

    # scan_name = self.scan_names[idx]        
    mesh_vertices = np.load(os.path.join(data_path, scan_name)+'_vert.npy')
    instance_labels = np.load(os.path.join(data_path, scan_name)+'_ins_label.npy')
    semantic_labels = np.load(os.path.join(data_path, scan_name)+'_sem_label.npy')
    instance_bboxes = np.load(os.path.join(data_path, scan_name)+'_bbox.npy')

    # ------------------------------- LABELS ------------------------------        
    target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
    target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
    angle_classes = np.zeros((MAX_NUM_OBJ,))
    angle_residuals = np.zeros((MAX_NUM_OBJ,))
    size_classes = np.zeros((MAX_NUM_OBJ,))
    size_residuals = np.zeros((MAX_NUM_OBJ, 3))

    # point_cloud, choices = pc_util.random_sampling(point_cloud,
    #     num_points, return_choices=True)        
    # instance_labels = instance_labels[choices]
    # semantic_labels = semantic_labels[choices]

    # pcl_color = pcl_color[choices]

    target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
    target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:6]


    class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:,-1]]   
    # NOTE: set size class as semantic class. Consider use size2class.
    size_classes[0:instance_bboxes.shape[0]] = class_ind
    size_residuals[0:instance_bboxes.shape[0], :] = \
        target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind,:]
        
    # ret_dict = {}
    # ret_dict['point_clouds'] = point_cloud.astype(np.float32)
    end_points['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
    end_points['heading_class_label'] = angle_classes.astype(np.int64)
    end_points['heading_residual_label'] = angle_residuals.astype(np.float32)
    end_points['size_class_label'] = size_classes.astype(np.int64)
    end_points['size_residual_label'] = size_residuals.astype(np.float32)
    target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))                                
    target_bboxes_semcls +=100
    target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
        [DC.nyu40id2class[x] for x in instance_bboxes[:,-1][0:instance_bboxes.shape[0]]]                
    end_points['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
    end_points['box_label_mask'] = target_bboxes_mask.astype(np.float32)
    end_points['mesh_vertice'] = mesh_vertices

    return end_points


if __name__=='__main__':
    num_input_channel = 1
    # Set file paths and dataset config
    demo_dir = os.path.join(ROOT_DIR, 'detection_vis')+"/"+FLAGS.model
    if not os.path.exists(demo_dir): os.mkdir(demo_dir) 
    
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import DC # dataset config
    # checkpoint_path = os.path.join(demo_dir, 'pretrained_mlcvnet_on_scannet.tar')
    checkpoint_path = FLAGS.checkpoint_path
    # Make sure your scannet ply file get transformed first, use rotate_val_scans.py to transform
    # Put any scannet transformed ply file in the demo_files folder and put its file name here
    # Then run demo.py
    scene_pc = FLAGS.scene_name +"_vh_clean_2.ply"
    scene_meta = FLAGS.scene_name +".txt"
    pc_path = os.path.join("scannet/scans/"+FLAGS.scene_name, scene_pc)

    #旋转原始场景点云并保存
    dump_dir = os.path.join(demo_dir, FLAGS.scene_name)
    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    clean_pc_path = FLAGS.pc_root+FLAGS.scene_name +"/"+ scene_pc
    scene_meta_path = FLAGS.pc_root+FLAGS.scene_name +"/"+scene_meta
    export_aligned_mesh(clean_pc_path,scene_meta_path,dump_dir+"/"+FLAGS.scene_name+".ply")
    print("Export axis aligned mesh")

    scan_name = FLAGS.scene_name
    
    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet_with_rn') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if FLAGS.model =='votenet_ARM3D':
        MODEL = importlib.import_module('ARM3D') # import network module
        net = MODEL.VoteNet_ARM3D(num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                num_proposal=FLAGS.num_target,
                input_feature_dim=num_input_channel,
                relation_pair=FLAGS.relation_pair,
                relation_type=FLAGS.relation_type,
                random=FLAGS.random).to(device)
        # print(net.state_dict().keys())

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
    pc_path = dump_dir+"/"+FLAGS.scene_name+".ply"
    # pc_path = '/home/lyq/Myproject/ARM3D_VIS/demo_files/scene0609_02_vh_clean_2.ply'
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
    end_points = get_GTlabel(end_points, scan_name)
    # print("KEYS:",end_points.keys())
    print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))
  

    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    MODEL.dump_results(end_points, dump_dir, DC, False)
    print('Dumped detection results to folder %s'%(dump_dir))
    




