# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:04:02 2023

@author: mfwos
"""

import pandas as pd
import numpy as np
import os
import torch
import albumentations as alb
import cv2
import copy
import torchvision.transforms as tv_trafos
from sklearn.model_selection import KFold
import datetime

if os.getcwd()[1:7] == 'kaggle':
    PROJECT_PATH = '/kaggle/input/plantation-counting-challenge/'
elif os.getcwd()[1:8] == 'content':
    PROJECT_PATH = '/content/drive/MyDrive/Digital Africa Plantation Counting Challenge/'
else: 
    PROJECT_PATH = 'C:/Users/mfwos/Documents/ML Projects/Digital Africa Plantation Counting Challenge/' # if os.getcwd()[0] == 'C' else ...

IMAGE_PATH = PROJECT_PATH + 'data/TreeImages/' 
script_time = datetime.datetime.now() - datetime.datetime(2023,1,1)

os.chdir(PROJECT_PATH)
from help_funcs import boost_model, f_base_model_prediction, f_base_model_train_one_epoch, base_model

train_csv = pd.read_csv(PROJECT_PATH + 'data/Train.csv') 
test_csv = pd.read_csv(PROJECT_PATH + 'data/Test.csv') 

image_ids = np.array(train_csv.ImageId)
targets = np.array(train_csv.Target)
test_img_ids = np.array(test_csv.ImageId)


'''
Define the global variables and required random states and do train-valid split
'''

global_seed = 123
split_seed = 13579 # keep the same to have the same splits
no_folds = 5
test_size = 1/no_folds
fold = 4
model_type = 'v1_b3'
pretrained = True
img_size = 224
batch_size = 8
max_epochs = 50
model_device = torch.device('cuda')
verbose = False

torch.manual_seed(global_seed**3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random_state = np.random.RandomState(global_seed)
splitter_state = np.random.RandomState(split_seed)

KFold_splitter = KFold(n_splits = no_folds, shuffle = True, random_state = splitter_state)
idxs_train, idxs_valid = list(KFold_splitter.split(image_ids))[fold]

train_size = len(idxs_train)
valid_size = len(idxs_valid)
no_batches = int(np.ceil(train_size/batch_size))
no_batches_valid = int(np.ceil(valid_size/batch_size))

'''
Define model_1
'''

net_version = 1 if model_type[:2] == 'v1' else 2
net_size = 'small' if model_type[-1] in ['3','s'] else 'medium'

model_1 = base_model(pretrained, net_version, net_size).to(model_device)

assert model_1.net_size == net_size and model_1.net_version == net_version, 'Model version and/or size do not fit'


'''
Get randomly generated training parameters
'''

lr = 0.00035
transformer = torch.nn.Sequential(
    tv_trafos.RandomApply(torch.nn.ModuleList([tv_trafos.GaussianBlur(5)]), p=0.2),
    tv_trafos.RandomHorizontalFlip(p = 0.5),
    tv_trafos.RandomVerticalFlip(p = 0.5),
    tv_trafos.RandomApply(torch.nn.ModuleList([tv_trafos.RandomRotation(45)]), p=0.3)
   )
transform_set = 2
early_stopping_crit = 6
#lr = f_get_random_lr(max_lr = 0.0007, min_lr = 0.0001, random_state = random_state)
#transformer, transform_set = f_get_random_transformer(random_state)
#early_stopping_crit = random_state.choice([5,6,7,8,9,10])

downsampler = alb.Resize(img_size, img_size, interpolation = cv2.INTER_LINEAR)

optimizer = torch.optim.AdamW(model_1.parameters(), lr = lr)
scheduler_mode = 'exponential'
#scheduler_mode = 'constant' if random_state.uniform() < 0.5 else 'exponential'

scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor = 1, total_iters = 0, last_epoch = -1) if scheduler_mode == 'constant' else torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

best_rmse = 9999
best_state_dict = copy.deepcopy(model_1.state_dict())
epochs_no_improv = 0
best_epoch = 0

for epoch in range(max_epochs):
    
    curr_rmse = f_base_model_train_one_epoch(model_1, scheduler, random_state, no_batches, batch_size, train_size,
                           idxs_train, downsampler, transformer, img_size, targets, no_batches_valid, valid_size,
                           idxs_valid, model_device, image_ids, IMAGE_PATH)

    if curr_rmse < best_rmse:
        best_rmse = curr_rmse
        best_state_dict = copy.deepcopy(model_1.state_dict())
        epochs_no_improv = 0
        best_epoch = epoch + 1
    else:
        epochs_no_improv += 1
    
    print("Finished epoch ", epoch," of training. RMSE this epoch = ", np.round(curr_rmse,5))
    
    if epochs_no_improv >= early_stopping_crit:
        print("No improvement after ", epochs_no_improv, " epochs. Training stopped.")
        break

model_1.load_state_dict(best_state_dict)
model_id = int(script_time.total_seconds())

if os.getcwd()[1:7] == 'kaggle':
    torch.save({
        'model_state_dict': model_1.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'valid_rmse': best_rmse,
        'epochs_trained': best_epoch,
        'model_id': model_id},'/kaggle/working/base_' + str(fold) + '.pth')
else: 
    torch.save({
        'model_state_dict': model_1.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'valid_rmse': best_rmse,
        'epochs_trained': best_epoch,
        'model_id': model_id},PROJECT_PATH + 'models_boosting/base_' + str(fold) + '.pth')

'''
Load first model 
'''

model_type = 'v1_b3'
pretrained = True

net_version = 1 if model_type[:2] == 'v1' else 2
net_size = 'small' if model_type[-1] in ['3','s'] else 'medium'
    
model_1 = base_model(pretrained, net_version, net_size).to(model_device)
model_1.load_state_dict(torch.load(PROJECT_PATH + 'models_boosting/base_' + str(fold) + '.pth')['model_state_dict'])
model_id = torch.load(PROJECT_PATH + 'models_boosting/base_' + str(fold) + '.pth')['model_id']
model_1.eval()

'''
Make predictions for first model and use them to modify targets
'''

downsampler = alb.Resize(img_size, img_size, interpolation = cv2.INTER_LINEAR)

model_1_preds = f_base_model_prediction (model_1, batch_size, image_ids, downsampler, None, img_size,
                             model_device, IMAGE_PATH)

boost_targets = targets - model_1_preds

'''
Define second model and its optimization variables
'''

model_2 = boost_model(pretrained).to(model_device)

optimizer = torch.optim.AdamW(model_2.parameters(), lr = lr)
scheduler_mode = 'exponential'
#scheduler_mode = 'constant' if random_state.uniform() < 0.5 else 'exponential'

scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor = 1, total_iters = 0, last_epoch = -1) if scheduler_mode == 'constant' else torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

best_rmse = 9999
best_state_dict = copy.deepcopy(model_2.state_dict())
epochs_no_improv = 0
best_epoch = 0

'''
Train model 2
'''

for epoch in range(max_epochs):
    
    curr_rmse = f_base_model_train_one_epoch(model_2, scheduler, random_state, no_batches, batch_size, train_size,
                           idxs_train, downsampler, transformer, img_size, boost_targets, no_batches_valid, valid_size,
                           idxs_valid, model_device, image_ids, IMAGE_PATH)

    if curr_rmse < best_rmse:
        best_rmse = curr_rmse
        best_state_dict = copy.deepcopy(model_2.state_dict())
        epochs_no_improv = 0
        best_epoch = epoch + 1
    else:
        epochs_no_improv += 1
    
    print("Finished epoch ", epoch," of training. RMSE this epoch = ", np.round(curr_rmse,5))
    
    if epochs_no_improv >= early_stopping_crit:
        print("No improvement after ", epochs_no_improv, " epochs. Training stopped.")
        break

model_2.load_state_dict(best_state_dict)
#model_id = int(script_time.total_seconds())

if os.getcwd()[1:7] == 'kaggle':
    torch.save({
        'model_state_dict': model_2.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'valid_rmse': best_rmse,
        'epochs_trained': best_epoch,
        'model_id': model_id},'/kaggle/working/boost_' + str(fold) + '.pth')
else: 
    torch.save({
        'model_state_dict': model_2.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'valid_rmse': best_rmse,
        'epochs_trained': best_epoch,
        'model_id': model_id},PROJECT_PATH + 'models_boosting/boost_' + str(fold) + '.pth')

