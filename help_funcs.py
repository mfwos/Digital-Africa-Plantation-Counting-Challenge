import imageio
import torchvision.models as tv_models 
import torch
import numpy as np


def f_read_image_from_png (img_names, IMG_PATH, downsampler = None, transformer = None, img_size = 1024, n_channels = 3):

    '''
    Reads the images in img_names from the specified path in IMG_PATH and returns
    a numpy array representing the images. 
    
    Args:
        img_names (list): List of names of the images to read.
        IMG_PATH  (str):  Path to the foldes containing the images.
        downsampler (albumentations.Resize object, optional): If the images
            should be downscaled, the downscaling object to be applied
        transformer (torchvision.transforms object, optional): If the images
            should be transformed (i.e. for augmentation), the transformation
            object to be applied
        img_size (int, optional):   Image size of output 
            (that is after a possible downsampling). Default = 1024
        n_channels (int, optional): number of channels of the images. Default = 3. 
    Output: Numpy array of shape (len(img_names), n_channels, img_size, img_size)
    '''
    
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
    '''
    Base model to count the plants within an image, using a
    pretrained EfficientNet B3 backbone.
    '''
    def __init__(self):
        '''
        Architecture of model:
            Image -> Backbone -> Pooling -> Linear Layer -> Scalar output (estimated count) 
        '''
        super(base_model, self).__init__()
        
        self.backbone = tv_models.efficientnet_b3(weights = 'EfficientNet_B3_Weights.DEFAULT')
        self.n_features = 1536 
        self.pooling_layer = torch.nn.AdaptiveAvgPool2d((1,1))
        self.final_layer = torch.nn.Linear(self.n_features, 1)
    
    
    def forward(self, image):
        '''
        Forward method to output an estimated plant count for each input image.
        Input:
            image (torch.tensor): image tensor of size (*,3,img_size, img_size)
        Output:
            torch.tensor of size (*) representing the estimated count of plants
            within each image of the input.
        '''
        x = self.backbone.features(image)
        x = self.pooling_layer(x).squeeze()
        x = self.final_layer(x)
            
        return x

        
def f_train_model_one_epoch(model, scheduler, shuffler, no_batches, batch_size, train_size,
                       idxs_train, downsampler, transformer, image_size, targets, no_batches_valid, valid_size,
                       idxs_valid, model_device, image_ids, IMAGE_PATH):
    '''
    Trains an instance of base_model or boost model for one epoch, including image reading, downsampling,
    and augmenting as well as updating the scheduler. After training of one epoch 
    the validation RMSE is calculated.
    
    Args:
        model (instance of class base_model or boost_model): The model to be trained.
        scheduler (torch.optim.scheduler): The training scheduler to be used during training.
        shuffler (np.random.RandomState): Numpy random state for reproducibility of shuffling of training indices.
        no_batches (int): number of batches in one epoch.
        batch_size (int): Size of one batch.
        train_size (int): Total number of training samples.
        idxs_train (list): Indices of images within the training set.
        downsampler (albumentations.Resize object): Downsampling Object to be applied 
            within the function f_read_image_from_png.
        transformer (torchvision.transforms object): Transformer Object to be applied
            within the function f_read_image_from_png.
        image_size (int): Size of the images (after downsampling)
        targets (numpy.array): Array of size (train_size) containing the targets. These
            need to be correctly alligned with the two index lists idxs_train and idxs_valid.
        no_batches_valid (int): Number of batches in one epoch of validation.
        valid_size (int): Total number of validation samples.
        idxs_valid (list): Indices of images within the validation set.
        model_device (torch.device object): Device on which the training should be done. 
            Must match the device of model.
        image_ids (list): List of all image ids, used to read the images from IMAGE_PATH.
            The order within in the list must align with the order of targets and the index
            lists idxs_train and idxs_valid.
        IMAGE_PATH (str): Name of path to the images.
    Output:
        Validation RMSE after training the model for one epoch.
    '''
    
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
    
    '''
    Upsampling model used to boost the prediction of a base_model instance.
    Again we use a pretrained a EfficientNet B3 backbone.
    '''
    
    def __init__(self):
        
        '''
        Architecture of model
            (I) image -> backbone part 1 -> (save features feats) 
                      -> backbone part 2 -> Upsample (factor x2) -> x
            (II) (merge x and feats) -> merged x
            (III) merged x -> Pooling -> Linear Layer -> Scalar output 
        
        '''
        super(boost_model, self).__init__()
        
        self.backbone = tv_models.efficientnet_b3(weights = 'EfficientNet_B3_Weights.DEFAULT')
        
        self.channel_reduction = torch.nn.Conv2d(1536,256,1)
        self.upsample = torch.nn.Upsample(scale_factor = 2, mode = 'bilinear')
        
        self.channel_merge = torch.nn.Conv2d(136,256,1)
        self.reg_gap = torch.nn.AdaptiveAvgPool2d((1,1))
        self.single_linear = torch.nn.Linear(256,1)
        
    def forward(self, image):
        '''
        Forward method to output an estimated boost for the count of each input image.
        Input:
            image (torch.tensor): image tensor of size (*,3,img_size, img_size)
        Output:
            torch.tensor of size (*) representing the estimated boost for the 
            count of plants within each image of the input.
        '''
        x = image
        for i in range(9):
            x = self.backbone.features[i](x) # The features of resolution 14x14 are saved
            if i == 5:
                features = x
        
        # After running through the backbone, the 7x7 feature map is linearly upsampled
        # to match a 14x14 resolution
        x = self.upsample(self.channel_reduction(x)) 
        
        # The saved features and the upsampled backbone-output are merged,
        # pooled and run through a final linear layer to generate the scalar output.
        x = self.channel_merge(features) + x
        x = self.single_linear(self.reg_gap(x).squeeze())
        
        return x 

def f_base_model_prediction (model, batch_size, image_ids, downsampler, transformer, image_size,
                             model_device, IMAGE_PATH):
    
    '''
    Calculates the predictions of an instance of base_model or boost model on a selected
    list of images. Equally to f_train_model_one_epoch the function 
    includes image reading and if necessary downsampling.
    
    Args:
        model (instance of class base_model or boost_model): The model to use for predictions.
        batch_size (int): Size of one batch.
        image_ids (list): List of all ids of images for which we would like to make predictions.
        downsampler (albumentations.Resize object): Downsampling Object to be applied 
            within the function f_read_image_from_png.
        transformer (torchvision.transforms object): Transformer Object to be applied
            within the function f_read_image_from_png.
        image_size (int): Size of the images (after downsampling).
        model_device (torch.device object): Device on which the training should be done. 
            Must match the device of model.
        IMAGE_PATH (str): Name of path to the images.
    Output:
        Numpy.array of size (len(img_ids)) representing the counting predictions for 
        the images in img_ids.
    '''
    
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
        
    