from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

from argparse import ArgumentParser


mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()
LOG_PARA = 100.0

dataRoot = 'test'

#model_path = 'exp/12-06_15-03_NWPU_Res101_SFCN_1e-05/SCAR-latest.pth'
model_path = 'test/SCAR-latest.pth'
model_name = "SCAR"


def test(imgname, model_path):

    net = CrowdCounter(cfg.GPU_ID, model_name)
    net.cuda()
    #lastest_state = torch.load(model_path)
    #net.load_state_dict(lastest_state['net'])
    net.load_state_dict(torch.load(model_path))
    net.eval()

    img = Image.open(imgname)
    if img.mode == 'L':
        img = img.convert('RGB')
    img = img_transform(img)[None, :, :, :]
    with torch.no_grad():
        img = Variable(img).cuda()
        crop_imgs, crop_masks = [], []
        b, c, h, w = img.shape
        rh, rw = 576, 768
        for i in range(0, h, rh):
            gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                mask = torch.zeros(b, 1, h, w).cuda()
                mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                crop_masks.append(mask)
        crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

        # forward may need repeatng
        crop_preds = []
        nz, bz = crop_imgs.size(0), 1
        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i+bz)
            crop_pred = net.test_forward(crop_imgs[gs:gt])
            crop_preds.append(crop_pred)
        crop_preds = torch.cat(crop_preds, dim=0)

        # splice them to the original size
        idx = 0
        pred_map = torch.zeros(b, 1, h, w).cuda()
        for i in range(0, h, rh):
            gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                idx += 1

        # for the overlapping area, compute average value
        mask = crop_masks.sum(dim=0).unsqueeze(0)
        pred_map = pred_map / mask

    pred_map = pred_map.cpu().data.numpy()[0,0,:,:]


    pred = np.sum(pred_map) / LOG_PARA

    print("Prediction value: ", int(pred))

if __name__ == '__main__':
    ap = ArgumentParser(description='The parameters for training.')
    ap.add_argument('--img_loc', type=str, default="test_image.jpg", help="Location for file to pass into model.") 
    ap.add_argument('--model_loc', type=str, default="SCAR-latest.pth", help="Location of SCAR model.") 
    ap = ap.parse_args()
    test(ap.img_loc, ap.model_loc)

    #python test_one_image.py --img_loc test_image.jpg --model_loc SCAR-latest.pth


