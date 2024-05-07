import torch
import cv2
import os
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset , DataLoader

class cityDataset(Dataset):
    
    def __init__(self , data_dir , train = True):
        
        self.raw_file_list = glob.glob(os.path.join(data_dir,'*.jpg'))
        self.train = train
        
        
        
        
        self.input_transform = A.Compose([
            
            A.Resize(286,286),
            A.ColorJitter(p=0.1),
            A.RandomCrop(256, 256),
            A.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
            ToTensorV2()
            
        ])
        
        self.train_val_target_transform = A.Compose([
            
            A.Resize(256,256),
            A.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
            ToTensorV2()
            
        ])
        
        
          
        
    def __len__(self):
        
        return len(self.raw_file_list)
               
        
        
        
    def __getitem__(self,index):
       
        img = cv2.imread(self.raw_file_list[index])
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        input_img = rgb_img[:,:rgb_img.shape[1]//2,:]
        target_img = rgb_img[:,rgb_img.shape[1]//2:,:]
        
        if self.train:
           
            input_tensor = self.input_transform(image=input_img)
            transformed_input_tensor  = input_tensor['image']
            target_tensor = self.train_val_target_transform(image=target_img)
            transformed_target_tensor  = target_tensor['image']
            
        else:
            
            input_tensor = self.train_val_target_transform(image=input_img)
            transformed_input_tensor  = input_tensor['image']
            target_tensor = self.train_val_target_transform(image=target_img)
            transformed_target_tensor  = target_tensor['image']
               
        return transformed_input_tensor, transformed_target_tensor
    
    
    

       
       
       
       
       
       
       


       