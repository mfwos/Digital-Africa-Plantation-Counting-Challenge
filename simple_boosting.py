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


'''
Define the project and data path
and load help functions
'''

PROJECT_PATH = '...'  # Set your path to the project directory here 

IMAGE_PATH = PROJECT_PATH + '/data/TreeImages/' 
script_time = datetime.datetime.now() - datetime.datetime(2023,1,1) # Script time will be used as unique model identifier

os.chdir(PROJECT_PATH)
from help_funcs import boost_model, f_base_model_prediction, f_train_model_one_epoch, base_model

'''
Load metadata of training data 
'''

train_csv = pd.read_csv(PROJECT_PATH + '/data/Train.csv') 

image_ids = np.array(train_csv.ImageId)
targets = np.array(train_csv.Target)


'''
Define the global variables and required random states and do train-valid split
'''

#### Define the fold that should be run 
no_folds = 5
fold = 0
######


global_seed = 123                   # random seeds for reproducibility
split_seed = 13579                  # keep the same to have the same splits
test_size = 1/no_folds
img_size = 224
batch_size = 8
max_epochs = 50                     # max number of epochs to train the models
model_device = torch.device('cuda') # chosse pytorch device, set 'cpu' if not 'cuda'-compatible 
verbose = False

# Set random states for pytorch and sklearn/numpy operations
torch.manual_seed(global_seed**3)           
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False
random_state = np.random.RandomState(global_seed)
splitter_state = np.random.RandomState(split_seed)

# Do the training/valid set split using KFold with fixed random state for 
# reproducibility of the splitted folds 
KFold_splitter = KFold(n_splits = no_folds, shuffle = True, random_state = splitter_state)
idxs_train, idxs_valid = list(KFold_splitter.split(image_ids))[fold]

train_size = len(idxs_train)
valid_size = len(idxs_valid)
no_batches = int(np.ceil(train_size/batch_size))
no_batches_valid = int(np.ceil(valid_size/batch_size))


'''
Define the base model as well as training parameters and 
a transformer for data input augmentation; 
see help_funcs for source code of base_model
'''

model_1 = base_model().to(model_device) # Base model

# Transformer for input augmentation: Random sequence of Gaussian Blurs,
# Horizontal and/or Vertical Flips and rotations applied to the input image
transformer = torch.nn.Sequential(
    tv_trafos.RandomApply(torch.nn.ModuleList([tv_trafos.GaussianBlur(5)]), p=0.2),
    tv_trafos.RandomHorizontalFlip(p = 0.5),
    tv_trafos.RandomVerticalFlip(p = 0.5),
    tv_trafos.RandomApply(torch.nn.ModuleList([tv_trafos.RandomRotation(45)]), p=0.3)
   )

# Apart from augmenting, we also need to downsample the images for faster computation
downsampler = alb.Resize(img_size, img_size, interpolation = cv2.INTER_LINEAR)

lr = 0.00035                # Learning Rate
early_stopping_crit = 6     # Early stopping: After 6 epochs without improvement we want to stop training

# Define optimizer and learning rate scheduler; we use an exponential decay for 
# our learning rate
optimizer = torch.optim.AdamW(model_1.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

# Initialize variables to document the training results
best_rmse = 9999
best_state_dict = copy.deepcopy(model_1.state_dict())
epochs_no_improv = 0
best_epoch = 0

'''
Training of base model
'''

# Start training for a maximum amount of max_epochs. The function
# f_train_model_one_epoch trains the model for one epoch, this
# includes data augmentation, downsampling and updating of weights and
# learning rate according to scheduler; see help_funcs for details
# After each epoch we check whether performance has been improved or not.
# After six epochs without improvement we stop. The best performing model is saved.
for epoch in range(max_epochs):
    
    curr_rmse = f_train_model_one_epoch(model_1, scheduler, random_state, no_batches, batch_size, train_size,
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

# Load the best model and save model as well as model_id and training results 
model_1.load_state_dict(best_state_dict)
model_id = int(script_time.total_seconds())


torch.save({
    'model_state_dict': model_1.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'valid_rmse': best_rmse,
    'epochs_trained': best_epoch,
    'model_id': model_id},PROJECT_PATH + '/models_boosting/base_' + str(fold) + '.pth')



'''
Now construct the boosting model;
to do so first we load the base model and make predictions using the base model,
in order to define the target for our boosting model
'''

# Load model    
model_1 = base_model().to(model_device)
model_1.load_state_dict(torch.load(PROJECT_PATH + '/models_boosting/base_' + str(fold) + '.pth')['model_state_dict'])
model_id = torch.load(PROJECT_PATH + '/models_boosting/base_' + str(fold) + '.pth')['model_id']
model_1.eval()

# Make base model predictions on the whole set; for details see help_funcs
model_1_preds = f_base_model_prediction (model_1, batch_size, image_ids, downsampler, None, img_size,
                             model_device, IMAGE_PATH)

# Define the targets for our boosting model; our final model will predict 
# targets according to 
# predicted target = predicted base target + predicted boosting target
# Thus our training targets for the boosting model are defined as the 
# difference between actual target value and the base model predictions.
boost_targets = targets - model_1_preds

'''
Define boosting model and its training parameters; for details of the 
boosting model see help_funcs
'''

model_2 = boost_model().to(model_device) # Boosting model

# Define optimizer and scheduler for training. We use the same learning rate
# and we also use an exponentially decaying learning rate we used with the base model
optimizer = torch.optim.AdamW(model_2.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

# Initialize variables to document the training results
best_rmse = 9999
best_state_dict = copy.deepcopy(model_2.state_dict())
epochs_no_improv = 0
best_epoch = 0

'''
Training of boosting model
'''

# Start training. The parameters of training are the same we used for
# the base model, except for the targets that we manipulated beforehand.
# For details see help_funcs.
for epoch in range(max_epochs):
    
    curr_rmse = f_train_model_one_epoch(model_2, scheduler, random_state, no_batches, batch_size, train_size,
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

# Load the best performing model and save model and training results 
model_2.load_state_dict(best_state_dict)

torch.save({
    'model_state_dict': model_2.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'valid_rmse': best_rmse,
    'epochs_trained': best_epoch,
    'model_id': model_id},PROJECT_PATH + '/models_boosting/boost_' + str(fold) + '.pth')

