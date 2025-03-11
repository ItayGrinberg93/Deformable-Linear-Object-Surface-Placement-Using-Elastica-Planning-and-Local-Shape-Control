import os
import numpy as np
import shutil
import random
from tqdm import tqdm

# creating train / val /test
data_path = "/Path_To_New_Data/" # path of source folder

root_dir = "/Path_To_Data/" # folder to copy images from

new_root = 'Folder_Name'


os.makedirs(data_path + new_root + '_train/')
os.makedirs(data_path + new_root +'_valid/')
os.makedirs(data_path + new_root + '_test/')
    
## creating partition of the data after shuffeling


src = root_dir  # folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)

## For example, Here : 
## training ratio = 0.75 ,
## test ratio = (0.95-0.75) [its "rain_ratio(0.75) + test_ration(0.2) = 0.95],
## and, no need to add validation ration, as it is automatically calculated
## validation ratio =  (1-0.95)

train_FileNames,val_FileNames,test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.7) ,int(len(allFileNames)*0.9)])

## Converting file names from array to list

train_FileNames = [src+ name for name in train_FileNames]
val_FileNames = [src+ name for name in val_FileNames]
test_FileNames = [src+name for name in test_FileNames]

## Copy pasting images to target directory

for name in tqdm(train_FileNames):
    shutil.copy(name, data_path + new_root+'_train/' )

for name in tqdm(val_FileNames):
    shutil.copy(name, data_path +new_root+'_valid/' )

for name in tqdm(test_FileNames):
    shutil.copy(name,data_path + new_root+'_test/' )










