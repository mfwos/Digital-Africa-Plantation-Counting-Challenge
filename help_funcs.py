# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:15:08 2023

@author: mfwos
"""

import imageio
import torchvision.models as tv_models 
import torch
import numpy as np


def f_read_image_from_png (img_names, IMG_PATH, downsampler = None, transformer = None, img_size = 1024, n_channels = 3):
    
    x = np.zeros((len(img_names), img_size, img_size, n_channels), dtype = np.float32) 
    
    for i, file_name in enumerate(img_names):
        
        new_img = imageio.v3.imread(IMG_PATH + file_name)
        if downsampler != None:
            new_img = downsampler(image = new_img)['image']
        x[i,:] = new_img
    
    x = torch.tensor(x, dtype = torch.float32).permute(0,3,1,2)
    x_trafo = transformer(x) if transformer != None else x 
    x_norm = x_trafo / 255
    
    return x_norm


class base_model (torch.nn.Module):
    
    def __init__(self, pretrained, net_version = 1, net_size = 'small'):
        
        super(base_model, self).__init__()
        self.pretrained = pretrained
        self.net_size = net_size
        self.net_version = net_version
        
        if net_version == 1 and net_size == 'small':
            
            self.model_name = 'v1_b3'
            if pretrained:
                self.backbone = tv_models.efficientnet_b3(weights = 'EfficientNet_B3_Weights.DEFAULT')
            else:
                self.backbone = tv_models.efficientnet_b3()
        
        if net_version == 2 and net_size == 'small':
            
            self.model_name = 'v2_s'
            if pretrained:
                self.backbone = tv_models.efficientnet_v2_s(weights = 'EfficientNet_V2_S_Weights.DEFAULT')
            else:
                self.backbone = tv_models.efficientnet_v2_s()
        
        if net_version == 1 and net_size == 'medium':
            
            self.model_name = 'v1_b5'
            if pretrained:
                self.backbone = tv_models.efficientnet_b5(weights = 'EfficientNet_B5_Weights.DEFAULT')
            else:
                self.backbone = tv_models.efficientnet_b5()
        
        if net_version == 2 and net_size == 'medium':
            
            self.model_name = 'v2_m'
            if pretrained:
                self.backbone = tv_models.efficientnet_v2_m(weights = 'EfficientNet_V2_M_Weights.DEFAULT')
            else:
                self.backbone = tv_models.efficientnet_v2_m()
        
        
        self.n_features = 1536 if self.model_name == 'v1_b3' else 1280 if self.model_name in ['v2_s','v2_m'] else 2048 
        self.pooling_layer = torch.nn.AdaptiveAvgPool2d((1,1))
        self.final_layer = torch.nn.Linear(self.n_features, 1)
        
    
    def forward(self, image):

        x = self.backbone.features(image)
        x = self.pooling_layer(x).squeeze()
        x = self.final_layer(x)
            
        return x

        
def f_base_model_train_one_epoch(model, scheduler, shuffler, no_batches, batch_size, train_size,
                       idxs_train, downsampler, transformer, image_size, targets, no_batches_valid, valid_size,
                       idxs_valid, model_device, image_ids, IMAGE_PATH):
        
    shuffler.shuffle(idxs_train)
        
    for i in range(no_batches):
        
        curr_batch_size = batch_size if i < no_batches - 1 else train_size - (batch_size * (no_batches - 1))
        curr_x = f_read_image_from_png(image_ids[idxs_train[(batch_size * i) : (batch_size * i + curr_batch_size)]], IMAGE_PATH, downsampler = downsampler, transformer = transformer, img_size = image_size)
        curr_y = targets[idxs_train[(batch_size * i) : (batch_size * i + curr_batch_size)]]
        
        output = model(curr_x.to(model_device)).squeeze()
        loss = torch.sqrt(torch.nn.MSELoss(reduction = 'sum')(output, torch.tensor(curr_y, dtype = torch.float32).to(model_device)) / batch_size)
        
        scheduler.optimizer.zero_grad()
        loss.backward()
        scheduler.optimizer.step()
    
    scheduler.step()
    model.eval()
    
    all_outputs = np.ones((len(idxs_valid))) * np.nan 
    
    for i in range(no_batches_valid):
        
        curr_batch_size = batch_size if i < no_batches_valid - 1 else valid_size - (batch_size * (no_batches_valid - 1))
        curr_x = f_read_image_from_png(image_ids[idxs_valid[(batch_size * i) : (batch_size * i + curr_batch_size)]], IMAGE_PATH, downsampler = downsampler, transformer = None, img_size = image_size)
        
        with torch.no_grad():
            output = model(curr_x.to(model_device)).squeeze()
        
        all_outputs[batch_size*i : (batch_size*i + curr_batch_size)] = output.detach().cpu().clone().numpy()
    

    curr_rmse = np.sqrt( ((all_outputs - targets[idxs_valid])**2).mean())
    
    model.train()
    
    return curr_rmse

class boost_model (torch.nn.Module):
    
    def __init__(self, pretrained):
        super(boost_model, self).__init__()
        
        if not pretrained:
            self.backbone = tv_models.efficientnet_b3()
        else:
            self.backbone = tv_models.efficientnet_b3(weights = 'EfficientNet_B3_Weights.DEFAULT')
        
        self.channel_reduction = torch.nn.Conv2d(1536,256,1)
        self.upsample = torch.nn.Upsample(scale_factor = 2, mode = 'bilinear')
        
        self.channel_merge = torch.nn.Conv2d(136,256,1)
        self.reg_gap = torch.nn.AdaptiveAvgPool2d((1,1))
        self.single_linear = torch.nn.Linear(256,1)
        
    def forward(self, image):
        
        x = image
        for i in range(9):
            x = self.backbone.features[i](x)
            if i == 5:
                features = x
        
        x = self.upsample(self.channel_reduction(x))
        #print(x.mean())
        #print(features.mean())
        #print(x.shape)
        x = self.channel_merge(features) + x
        #print(x.shape)
        x = self.single_linear(self.reg_gap(x).squeeze())
        
        return x 

def f_base_model_prediction (model, batch_size, image_ids, downsampler, transformer, image_size,
                             model_device, IMAGE_PATH):
    
    is_training = model.training
    model.eval()
    sample_size = len(image_ids)
    no_batches = int(np.ceil(len(image_ids) / batch_size))
    
    all_outputs = np.zeros((sample_size)) * np.nan
    
    for i in range(no_batches):
        
        curr_batch_size = batch_size if i < no_batches - 1 else sample_size - (batch_size * (no_batches - 1))
        curr_x = f_read_image_from_png(image_ids[(batch_size * i) : (batch_size * i + curr_batch_size)], IMAGE_PATH, downsampler = downsampler, transformer = transformer, img_size = image_size)
        
        output = model(curr_x.to(model_device)).squeeze()
        all_outputs[batch_size*i : (batch_size*i + curr_batch_size)] = output.detach().cpu().clone().numpy()
    
    if is_training:
        model.train()
    
    return all_outputs
        
    