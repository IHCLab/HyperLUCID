"""
HyperLUCID Utility Functions

Code implemented by:
    Shih-Min Hsu
    Ching-Yun Liang
"""
import os
import argparse
import numpy as np
from sklearn.metrics import *
import torch
import time
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torchmetrics.image import *
from tqdm import tqdm

import model
from functions import *

def main():
    parser = argparse.ArgumentParser(description='HCD')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--path-config', type=str, default='./config.yaml')
    parser.add_argument('-pc', '--print-config', action='store_true', default=False)
    parser.add_argument('-sr', '--show-results', action='store_true', default=False)
    parser.add_argument('--save-results', action='store_true', default=True)
    parser.add_argument('--dataset-path', type=str, default='./Data')
    args = parser.parse_args() 
    config = load_config(args.path_config)
    for key, value in config.items():
        globals()[key] = value
    
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)  
    np.random.seed(0)
    random.seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = False 
        
    x1, x2, bands, gt_reshape = load_data(args, data_name)

    if args.print_config:
        print(config)       
    
    x1=np.array(x1, np.float32)
    x1=torch.from_numpy(x1.astype(np.float32)).to(args.device)
    
    x2=np.array(x2, np.float32)
    x2=torch.from_numpy(x2.astype(np.float32)).to(args.device)
    
    iter = 0
    while True:
        if iter == 0:
            generate_mask_time = 0
            stopping_critetion_time = 0
            tic1 = time.time()
            rough_change, std, stop, sc_time = sample_selection(x1, x2, 1e-5, False, iter) 
            stopping_critetion_time += sc_time
            mask = (1 - rough_change).unsqueeze(-1) 
            toc1 = time.time()
            generate_mask_time += toc1 - tic1
        else:
            tic1 = time.time()
            x1 = x1_upgraded
            rough_change = predict
            mask = mask| (1 - rough_change).unsqueeze(-1) 
            toc1 = time.time()
            generate_mask_time += toc1 - tic1 + updated_sam_time

        # model
        tic2 = time.time()
        if iter == 0:
            training_time = 0
            net = model.Calibration_Function(int(bands/2)).to(args.device) 

        # train
        print("\n\n==================== training iteration {} ====================\n".format(iter))
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        criterion = nn.L1Loss()
        for i in tqdm(range(int(max_epoch * 0.7**iter))):
            net.train()
            optimizer.zero_grad()
            output= net(x1)
            loss = criterion(output * mask, x2 * mask)  
            loss.backward(retain_graph=False)
            optimizer.step()  
            scheduler.step()
            torch.cuda.empty_cache()
            if i % 5 == 0:
                print("{}\tLoss={:.4f}".format(str(i + 1), loss))
        toc2 = time.time()
        training_time += toc2 - tic2
        
        # test
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            net.eval()
            filename='./model_{}.pt'.format(data_name)
            torch.save(net.state_dict(), filename)
            print('save model...')

            if iter == 0:
                update_time = 0
            tic3 = time.time()
            x1_upgraded = net(x1) 
            toc3 = time.time()
            update_time += toc3 - tic3

            tic4 = time.time()
            predict, std, stop, sc_time = sample_selection(x1_upgraded, x2, std, stop, iter)
            toc4 = time.time()
            updated_sam_time = toc4 - tic4
            stopping_critetion_time += sc_time

        iter += 1
        
        if stop:
            accuracy_assessment(gt_reshape, np.reshape(predict.cpu().numpy()+1, [-1]))
            print('Alltime=', generate_mask_time + training_time + update_time)
            break


    plt.figure()
    plt.imshow(predict.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.axis("off")
    filename='./change_map_{}.png'.format(data_name)
    plt.savefig(os.path.join(save_path, filename), dpi = 300, bbox_inches='tight', pad_inches=0)
    plt.close() 

if __name__ == '__main__':
    main()

