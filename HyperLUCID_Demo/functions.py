"""
HyperLUCID Utility Functions

Code implemented by:
    Shih-Min Hsu
    Ching-Yun Liang
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import torch
import yaml
import scipy.io as sio
from torchmetrics.image import *
from einops import rearrange
import numpy as np
import time

sam = SpectralAngleMapper(reduction='none')

def sample_selection(x1, x2, old_std, stop, iter):
    (h, w, c) = x1.shape
    x1_temp = rearrange(x1, 'h w c -> (h w) c')
    x2_temp = rearrange(x2, 'h w c -> (h w) c')
    x1_temp = torch.unsqueeze(torch.unsqueeze(x1_temp, -1), -1)
    x2_temp = torch.unsqueeze(torch.unsqueeze(x2_temp, -1), -1)
    spectral_map = sam(x1_temp, x2_temp)
    spectral_map = rearrange(spectral_map, 'L b1 b2-> L (b1 b2)')

    ratio = 0.8

    _, idx = torch.topk(spectral_map.flatten(), int(h * w * ratio), largest=True)

    change_map = torch.zeros_like(spectral_map, dtype=torch.int)
    change_map.view(-1)[idx] = 1  
    
    tic = time.time()
    mask = (1 - change_map).detach().cpu().numpy()
    new_std = np.std(spectral_map.detach().cpu().numpy()[mask == 1])
    
    if abs(new_std - old_std) / old_std <= 0.05 or iter == 9: 
        threshold = 0.2
        change_map = (spectral_map > threshold).int()
        stop = True
    toc = time.time()
    
    change_map = change_map.view(h, w)  
    stopping_critetion_time = toc - tic
    
    return change_map, new_std, stop, stopping_critetion_time 

def normalization(X):
    x, y, z = X.shape
    temp = X.reshape((x * y, z))
    mean = np.mean(temp, axis=0)
    temp1 = temp - mean
    var = np.sum(temp1**2, axis=0) / (x * y)
    std = np.sqrt(var)
    temp2 = temp1 / std
    out = temp2.reshape((x, y, z))
    return out

def load_config(path):
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


def load_data(args, data_name):
    if data_name=='Farm':
        
        x1=sio.loadmat(args.dataset_path+'/Farm1.mat')
        x1=x1["imgh"].astype(np.float32)
        x1 = (x1 - np.min(x1))/(np.max(x1) - np.min(x1))
        x2=sio.loadmat(args.dataset_path+'/Farm2.mat')
        x2=x2["imghl"].astype(np.float32)
        x2 = (x2 - np.min(x2))/(np.max(x2) - np.min(x2))
        data_gt=sio.loadmat(args.dataset_path+'/label.mat')
        data_gt=data_gt["label"].astype(np.float32)
        data_gt=data_gt.astype(int)
        data_gt=data_gt+1
        data=np.concatenate((x1, x2), axis=2)
        _, _, bands = data.shape
        gt_reshape = np.reshape(data_gt, [-1])
        return x1, x2, bands, gt_reshape
    
    else:
        raise ValueError("Unsupported data type")
    
def accuracy_assessment(img_gt, changed_map):
    '''
        assess accuracy of changed map based on ground truth
    '''
    esp = 1e-6

    cm = changed_map
    gt = img_gt

    conf_mat = confusion_matrix(y_true=gt, y_pred=cm)
    kappa_co = cohen_kappa_score(y1=gt, y2=cm)

    TN, FP, FN, TP = conf_mat.ravel()
    P = TP / (TP + FP + esp)
    R = TP / (TP + FN + esp)
    F1 = 2 * P * R / (P + R + esp)
    oa = (TP + TN) / (TP + TN + FP + FN + esp)
    
    print('OA=', oa)
    print('Kappa=', kappa_co)
    print('F1=', F1)
    print('Precision=', P)
    print('Recall=', R)
