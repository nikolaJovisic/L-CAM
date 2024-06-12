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
from utils.LoadData import data_loader, mammo_loader
from utils.Restore import restore
from models import VGG16_L_CAM_Fm, VGG16_L_CAM_Img, VGG16_7x7_L_CAM_Img, VGG16_L_CAM_ImgA



# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:',ROOT_DIR)
txt_path = r"C:\Users\Korisnik\Documents\GitHub\L-CAM\datalist\inbreast\train.txt"
IMG_DIR  = r'C:\Users\Korisnik\Documents\GitHub\mammography\data\INBREAST\AllDICOMs'

Snapshot_dir = os.path.join(ROOT_DIR,'snapshots', 'Mammo_VGG16_Img')

percent = 0
def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet50_aux_ResNet18_init')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--train_list", type=str, default=txt_path)
    parser.add_argument("--test_list", type=str, default=txt_path)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--arch", type=str,default='VGG16_L_CAM_Img')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--disp_interval", type=int, default=40)
    parser.add_argument("--snapshot_dir", type=str, default=Snapshot_dir)
    parser.add_argument("--restore_from", type=str, default= r"C:\Users\Korisnik\Downloads\imagenet_epoch_7_glo_step_2161.pth.tar")
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--percentage", type=float, default=percent)
    return parser.parse_args()



current_epoch=0
model=None
args = get_arguments()

top1 = AverageMeter()
top5 = AverageMeter()
top1.reset()
top5.reset()

top1_ = AverageMeter()
top5_ = AverageMeter()
top1_.reset()
top5_.reset()

def get_model(args):
    model = eval(args.arch).model()
    #model.cuda()
    model.eval()
    optimizer = my_optim.get_optimizer(args, model)
    if args.resume == 'True':
         restore(args, model, optimizer)
    return  model, optimizer


val_loader = mammo_loader(txt_path=txt_path,img_dir=IMG_DIR, batch_size=1)
global_counter = 0
prob = None
gt = None

if model is None:
    model, _ = get_model(args)

model.eval() 
#model.cuda()


y_mask_image = []
y_image = []

#model.cuda()

def denormalize(tensor):
    means, stds = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    denormalized=transforms.Normalize(-1*means/stds,1.0/stds)(tensor)
    return denormalized

k = 0
time_spend = 0
  
# Read Images 
for idx, dat  in tqdm(enumerate(val_loader)):
    img_path, img, label_in = dat
    im = img
    global_counter += 1
    label = label_in
    #img, label = img.cuda(), label.cuda()
    now = time.time()


    with torch.no_grad():
        logits = model(img,label,isTrain = False)
        logits0 = logits[0]        
    logits0 = F.softmax(logits0, dim=1)
    prec1_1, prec5_1 = Metrics.accuracy(logits0.cpu().data, label_in.long(), topk=(1,5))
    y = logits0.cpu().data.numpy()
    class_1 = logits0.max(1)[-1]  #
    index_gt_y = class_1.long().cpu().data.numpy()  #
    Y_i_c = logits0.max(1)[0].item()
    y_image.append(Y_i_c)
    top1.update(prec1_1[0], img.size()[0])
    top5.update(prec5_1[0], img.size()[0])

    import random
    cam_map = model.get_c(index_gt_y)
    cam_map = cam_map[0]
    cam_map = Metrics.normalizeMinMax(cam_map)  
    cam_map = F.interpolate(cam_map, size=(224,224), mode='bilinear', align_corners=False)
    cam_map =  Metrics.drop_Npercent(cam_map,args.percentage)
    image = img
    mask_image =  image  * cam_map

    with torch.no_grad():
        logits = model(mask_image,label,isTrain = False)
        logits0 = logits[0]
    logits0 = F.softmax(logits0, dim=1)
    prec1_1, prec5_1 = Metrics.accuracy(logits0.cpu().data, label_in.long(), topk=(1,5))
    y = logits0.cpu().data.numpy()
    Y_i_c_ = logits0[:,index_gt_y][0].item()
    y_mask_image.append(Y_i_c_)
    top1_.update(prec1_1[0], img.size()[0])
    top5_.update(prec5_1[0], img.size()[0])    

y_image = np.array(y_image)
y_mask_image = np.array(y_mask_image)

print('Average Drop is: ', Metrics.AD(y_image,y_mask_image))
print('Increase in confidence is: ', Metrics.IC(y_image,y_mask_image))
print('Top1:', top1.avg, 'Top5:',top5.avg)
print('Top1:', top1_.avg, 'Top5:',top5_.avg)
