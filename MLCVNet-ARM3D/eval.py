# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
import xlwt, xlrd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: scannet]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 40000]')
parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
parser.add_argument('--cluster_sampling', default='seed_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: seed_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
parser.add_argument('--gpu', type=int, default=0, help='gpu to allocate')
parser.add_argument('--obj_pred', type=int, default=1, help='whether to restore votenet to predict obj')
parser.add_argument('--RM3D', type=int, default=1, help='whether to restore votenet to predict obj')

FLAGS = parser.parse_args()

if FLAGS.use_cls_nms:
    assert(FLAGS.use_3d_nms)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG

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

if FLAGS.dataset == 'scannet':
    FLAGS.dump_dir = 'eval_scannet_final'
elif FLAGS.dataset == 'sunrgbd':
    FLAGS.dump_dir = 'eval_sunrgbd_final'

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point

DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
assert(CHECKPOINT_PATH is not None)
time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
CHECKPOINT_DIR = CHECKPOINT_PATH.split("/")[-2]
DUMP_DIR = os.path.join(DUMP_DIR,CHECKPOINT_DIR+"_"+time_string)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    DUMP_FOUT.write(out_str+'\n')
    DUMP_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT,
        augment=False, use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)
print(len(TEST_DATASET))
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
    shuffle=FLAGS.shuffle_dataset, num_workers=4, worker_init_fn=my_worker_init_fn)
print("FLAGS.shuffle_dataset:",FLAGS.shuffle_dataset)
 
# Init the model and optimzier
MODEL = importlib.import_module('mlcvnet') # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

if FLAGS.obj_pred ==1:
    if FLAGS.RM3D ==1:
        Detector = MODEL.MLCVNet_3DRM
        print("MODEL.MLCVNet_3DRM")
    else:
        Detector = MODEL.MLCVNet_pred
        print("MODEL.MLCVNet_pred")
else:
    Detector = MODEL.MLCVNet
    print("MODEL.MLCVNet")

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling,
               relation_pair=FLAGS.relation_pair,
               relation_type=FLAGS.relation_type,
               random=FLAGS.random)
net.to(device)
if FLAGS.obj_pred ==1:
    criterion = MODEL.get_loss_with_rn
else:
    criterion = MODEL.get_loss
    
# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint if there is any
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log_string("Loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, epoch))

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms, 'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh, 'dataset_config':DATASET_CONFIG}
# ------------------------------------------------------------------------- GLOBAL CONFIG END

def evaluate_one_epoch():
    stat_dict = {}
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds'],'center_label':batch_data_label['center_label']}
        with torch.no_grad():
            end_points = net(inputs)


        end_points['relation_type'] = FLAGS.relation_type
        end_points['relation_pair'] = FLAGS.relation_pair

        if FLAGS.dataset == 'sunrgbd':
            end_points['model']= 'sunrgbd'
        if FLAGS.dataset == 'scannet':
            end_points['model']= 'scannet'
            
        # Compute loss
        for key in batch_data_label:
            if key != 'center_label': 
                assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    class_list = ['window', 'bed', 'counter', 'sofa', 'table', 'showercurtrain', 'garbagebin', 'sink', 'picture',
                  'chair', 'desk', 'curtain', 'refrigerator', 'door', 'toilet', 'bookshelf', 'bathtub', 'cabinet',
                  'mAP', 'AR']
    class_list_sunrgbd =  {'bed','table','sofa','chair', 'toilet','desk','dresser','night_stand', 'bookshelf','bathtub','mAP', 'AR'}
            
    baseline = {
        'window': [0.3729, 0.0789],
        'bed': [0.8744, 0.7670],
        'counter': [0.6258, 0.2011],
        'sofa': [0.8954, 0.6904],
        'table': [0.5792, 0.4180],
        'showercurtrain': [0.6705, 0.0775],
        'garbagebin': [0.3876, 0.1405],
        'sink': [0.4824, 0.2106],
        'picture': [0.0656, 0.0076],
        'chair': [0.8857, 0.6730],
        'desk': [0.6681, 0.3252],
        'curtain': [0.4133, 0.1058],
        'refrigerator': [0.4842, 0.2889],
        'door': [0.4718, 0.1468],
        'toilet': [0.9733, 0.8207],
        'bookshelf': [0.4902, 0.2786],
        'bathtub': [0.8933, 0.7931],
        'cabinet': [0.3886, 0.0936],
        'mAP': [0.5901, 0.3399]
    }
    baseline_sunrgbd = {
        'bed': [0.858, 0],
        'table': [0.504, 0],
        'sofa': [0.663, 0],
        'chair': [0.758,0],
        'toilet': [0.5792, 0],
        'desk': [0.265, 0],
        'dresser': [0.313, 0],
        'night_stand': [0.615, 0],
        'bookshelf': [0.319, 0],
        'bathtub': [0.792, 0],
        'mAP':[0,0]
    }
    
    # write the results to the excel
    workbook = xlwt.Workbook()
    style1 = xlwt.XFStyle()
    al = xlwt.Alignment()
    al.horz = 0x02
    al.vert = 0x01
    style1.alignment = al

    style2 = xlwt.XFStyle()
    font = xlwt.Font()
    font.bold = True  # 黑体
    style2.font = font
    style2.alignment = al
    
    if FLAGS.dataset =='sunrgbd':
        class_list = class_list_sunrgbd
        baseline = baseline_sunrgbd

    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
        metrics_dict = ap_calculator.compute_metrics()
        
        worksheet = workbook.add_sheet('eval_results_{0:02d}'.format(int(AP_IOU_THRESHOLDS[i]*100)))
        worksheet.write(0, 0, AP_IOU_THRESHOLDS[i], style1)
        worksheet.write(1, 0, "baseline", style1)
        worksheet.write(2, 0, "adaptive", style1)
        
        cols_num = 0
        for cls in class_list:
            for key in metrics_dict:
                if cls in key:
                    log_string('eval %s: %f'%(key, metrics_dict[key]))

                    # write to excel worksheet
                    if 'Recall' not in key and 'AR' not in key:
                        cols_num += 1
                        worksheet.write(0, cols_num, cls, style1)

                        if metrics_dict[key]>baseline[cls][i]:
                            worksheet.write(1, cols_num, baseline[cls][i], style1)
                            worksheet.write(2, cols_num, metrics_dict[key], style2)
                        else:
                            worksheet.write(1, cols_num, baseline[cls][i], style2)
                            worksheet.write(2, cols_num, metrics_dict[key], style1)

        #for key in metrics_dict:
        #    log_string('eval %s: %f'%(key, metrics_dict[key]))
    # for i, ap_calculator in enumerate(ap_calculator_list):
    #     print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
    #     metrics_dict = ap_calculator.compute_metrics()
    #
    #     for cls in class_list:
    #         for key in metrics_dict:
    #             if cls in key:
    #                 log_string('eval %s: %f'%(key, metrics_dict[key]))

    # workbook.save("./log_scannet/eval_" + FLAGS.checkpoint_path.split('/')[1] + ".xls")
    workbook.save(DUMP_DIR+"/eval_map.xls")

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss

def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = evaluate_one_epoch()

if __name__=='__main__':
    eval()
