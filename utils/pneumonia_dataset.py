

import torch
import torch.utils.data
from imgaug import augmenters as iaa
import random
import numpy as np
import cv2
import pydicom

    
class PneumoniaDataset(torch.utils.data.Dataset):
    def __init__(self,images, dimsxy=64,train=True, masks=None, transform=True):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.train = train
        self.images = images
        self.transform = transform
        #default dimensions
        self.dim_x = dimsxy
        self.dim_y = dimsxy
        if self.train:
            self.masks = masks
            
    def transform_(self, image, mask):
#         print('transform')

        
        randy = random.randint(0,7)
            
        if randy == 0:
            pass #nothing, just a normal image
        
        #flip left to right only
        if randy == 1:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        #flip up or down only
        if randy ==2:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        #rotate 90 degrees
        if randy ==3:
            image = np.rot90(image).copy()
            mask = np.rot90(mask).copy()
            
        #flip lr and up
        if randy ==4:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
            
        if randy ==5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            image = np.rot90(image).copy()
            mask = np.rot90(mask).copy()
        
        if randy ==6:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
            image = np.rot90(image).copy()
            mask = np.rot90(mask).copy()
        
        
            
#
        #need to reshape here before using iaa augs
        image = image.reshape(1,self.dim_x, self.dim_y)
        mask = mask.reshape(1,self.dim_x, self.dim_y)
#         https://imgaug.readthedocs.io/en/latest/source/augmenters.html
        
        if self.transform:
            if random.random() > .8:
                aug = iaa.GaussianBlur(sigma=(1.0,2.0))
                image = aug.augment_images(image)
    
            if random.random() > .9:
                aug = iaa.PiecewiseAffine(scale=(.01,.06))
                image = aug.augment_images(image)
            
            
        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.train:
            mask = self.masks[idx]
            return self.transform_(image, mask)
        
        image = image.reshape(1,self.dim_x, self.dim_y)
        return image
    


class PneumoniaDataset_test(torch.utils.data.Dataset):
    def __init__(self,df, dims=256):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        
        self.df = df
        self.dims = dims      
            
    def getPatch(self,img_id):
        mod = 256
        count=0
        X_test = np.zeros((256, self.dims, self.dims), dtype=np.float32)
        
        img = pydicom.dcmread(self.df.iloc[img_id]['file_path']).pixel_array
        topBorderWidth = mod
        bottomBorderWidth = mod
        leftBorderWidth = mod
        rightBorderWidth= mod

        img = cv2.copyMakeBorder(
                     img, 
                     topBorderWidth, 
                     bottomBorderWidth, 
                     leftBorderWidth, 
                     rightBorderWidth, 
                     cv2.BORDER_REFLECT             
                  )
        
#         image(img)

        for i in range(256, 1280,64):
            for j in range( 256,1280,64): 
    #             print(i,j)
                X_test[count] = img[i-128:i+128, j-128:j+128]
                count+=1
            
#         print('dataset size: ', count)
        X_test /=255
#         t=rebuild_(X_test)
#         image(t)
        return X_test
            
    
     
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #should return 256 patches
        images = self.getPatch(idx)
        
        return images
        