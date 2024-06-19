import sys

sys.path.append('../')
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import time
import argparse
import os
from tqdm import tqdm
import numpy as np
from torchvision import models, transforms
import L_CAM_VGG16.my_optim as my_optim
import torch.nn.functional as F
from utils.avgMeter import AverageMeter
from utils import Metrics
from utils.LoadData import data_loader, mammo_loader, inference_loader
from utils.Restore import restore
from models import Mammo_VGG16_Img
import matplotlib.pyplot as plt
import cv2
from preprocess import preprocess_scan, reverse_spatial_changes

# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:', ROOT_DIR)
IMG_DIR = r'C:\Users\Korisnik\Desktop\f100'

Snapshot_dir = os.path.join(ROOT_DIR, 'snapshots', 'Mammo_VGG16_Img')

percent = 0

def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet50_aux_ResNet18_init')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--train_list", type=str, default='')
    parser.add_argument("--test_list", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--arch", type=str, default='Mammo_VGG16_Img')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--disp_interval", type=int, default=40)
    parser.add_argument("--snapshot_dir", type=str, default=Snapshot_dir)
    parser.add_argument("--restore_from", type=str,
                        default=r"C:\Users\Korisnik\Documents\GitHub\L-CAM\snapshots\Mammo_VGG16_Img\imagenet_epoch_7_glo_step_2161.pth.tar")
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--percentage", type=float, default=percent)
    return parser.parse_args()


current_epoch = 0
args = get_arguments()

def get_model(args):
    model = eval(args.arch).model()
    model.eval()
    optimizer = my_optim.get_optimizer(args, model)
    if args.resume == 'True':
        restore(args, model, optimizer)
    return model, optimizer


val_loader = inference_loader(img_dir=IMG_DIR, batch_size=1)
model, _ = get_model(args)
model.eval()


def process_img(image):
    with torch.no_grad():
        logits = model(image, 1, isTrain=False)
        probability = logits[0][0][1].item()

    cam_map = model.get_c([0])
    cam_map = cam_map[0]
    cam_map = Metrics.normalizeMinMax(cam_map)
    cam_map = F.interpolate(cam_map, size=(1152, 896), mode='bilinear', align_corners=False)
    cam_map = Metrics.drop_Npercent(cam_map, args.percentage)

    s_image = np.transpose(image[0].cpu().numpy().astype(np.uint8), (1, 2, 0))
    s_mask = np.transpose((cam_map[0].cpu().numpy()), (1, 2, 0))

    channel0 = s_image[..., 0]
    channel1 = s_image[..., 1]
    modified_channel0 = channel0 * s_mask[..., 0]
    modified_channell = channel1 * s_mask[..., 0]
    s_image[..., 0] = modified_channel0
    s_image[..., 1] = modified_channell

    return probability, s_image

# Read Images
for idx, dat in tqdm(enumerate(val_loader)):
    img_path, raw_img, clean_img, spatial_changes = dat

    spatial_changes[0] = [i.numpy()[0] for i in spatial_changes[0]]
    spatial_changes[1] = spatial_changes[1].numpy()[0]
    spatial_changes[2] = [i.numpy()[0] for i in spatial_changes[2]]
    spatial_changes[3] = [i.numpy()[0] for i in spatial_changes[3]]

    spatial_changes[0].extend((0, 0))

    probability, heatmap = process_img(clean_img)
    reverted_heatmap = reverse_spatial_changes(heatmap, spatial_changes)
    print()
    # print(probability)
    # cv2.imshow('map', heatmap)
    # cv2.waitKey()


