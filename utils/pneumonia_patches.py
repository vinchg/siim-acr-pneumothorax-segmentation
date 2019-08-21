

import torch
import torch.utils.data
from imgaug import augmenters as iaa
import random
import numpy as np
import cv2
import pydicom
from scipy.sparse import csc_matrix, save_npz, load_npz
import scipy

    
class PneumoniaDataset(torch.utils.data.Dataset):
    def __init__(self,df, dims=1024, patchSz=256, miniBatch=1, 
                    train=True,val=False,transform=False):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.train = train
        self.val = val
        self.df = df
        self.transform = transform
        self.dims = dims
        self.patchSz = patchSz
        self.sz = miniBatch        

        
    def cropPad(self,imgs, lbls):
        augmenters_imgs = [
        iaa.CropAndPad(percent=(-0.5, 0.2)
        )]                           
        
        seq_imgs = iaa.Sequential(augmenters_imgs, random_order=False)        
        seq_imgs_deterministic = seq_imgs.to_deterministic()

        imgs_aug = seq_imgs_deterministic.augment_images(imgs)
        masks_aug = seq_imgs_deterministic.augment_images(lbls)
        return imgs_aug, masks_aug

    def affine(self,imgs, lbls):
        augmenters_imgs = [
        iaa.PiecewiseAffine(scale=(.01,.07)
        )]                           
        
        seq_imgs = iaa.Sequential(augmenters_imgs, random_order=False)        
        seq_imgs_deterministic = seq_imgs.to_deterministic()

        imgs_aug = seq_imgs_deterministic.augment_images(imgs)
        masks_aug = seq_imgs_deterministic.augment_images(lbls)
        return imgs_aug, masks_aug
       
            
    # def resize(self, idx):     
    #     img = pydicom.dcmread(self.df.iloc[idx]['file_path']).pixel_array
    #     img = scipy.misc.imresize(img, (self.dims,self.dims))

    #     if self.train:
    #         if self.df.iloc[idx]['has_pneumothorax']:
    #             mk = load_npz('siim/mask/'+self.df.iloc[idx]['id']+'.npz').todense().astype('uint8')
    #             mk[mk>0]=1
    #             mk = scipy.misc.imresize(mk,(self.dims,self.dims)).astype('uint8')
    #             return img/255, mk
    #         else:
    #             mk = np.zeros((self.dims, self.dims), dtype=np.uint8)
    #             return img/255, mk
    #     else:
    #         return img/255
    
    def getImage(self, idx):
        img = pydicom.dcmread(self.df.iloc[idx]['file_path']).pixel_array        
        if self.train:
            if self.df.iloc[idx]['has_pneumothorax']:
                mk = load_npz('siim/mask/'+self.df.iloc[idx]['id']+'.npz').todense().astype('uint8')
                mk[mk>0]=1
                return img/255, mk
            else:
                mk = np.zeros((self.dims, self.dims), dtype=np.uint8)
                return img/255, mk
        else:
            return img/255

    def sample(self):
        #the dataset was split where the first 1903 samples are pos and the rest are neg
        pos = 1903
        neg = 1904
        randy = random.randint(0,6)
        if randy==6:
            return random.randint(neg, len(self.df)-1)
        else: return random.randint(0,pos)

    def transform_(self, image, mask):
#         print('transform')        
        randy = random.randint(0,2)
            
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
            
#
        #need to reshape here before using iaa augs
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
#         https://imgaug.readthedocs.io/en/latest/source/augmenters.html
        
        if self.transform:
            randy = random.randint(0,3)
            if randy == 0:
                aug = iaa.GaussianBlur(sigma=(1.0,2.0))
                image = aug.augment_images(image)
    
            if randy == 1:
                aug = iaa.PiecewiseAffine(scale=(.01,.06))
                image = aug.augment_images(image)
            
            if randy == 2:
                image, mask = self.cropPad(image,mask)

            if randy == 3:
                pass           
            
        return image, mask


    def getPatches(self, img, mk):
        sz = self.sz
        if img.ndim==3:
            img = img.squeeze()
            mk = mk.squeeze()

        X_train = np.zeros((sz, self.patchSz, self.patchSz), dtype=np.float32)
        Y_train = np.zeros((sz, self.patchSz, self.patchSz), dtype=np.uint8)
        
        half = self.patchSz//2
        list_pos = np.argwhere(mk>0)
        if len(list_pos)>0:
            
            for i in range(sz):
                randy = np.random.randint(list_pos.shape[0])
                x,y = list_pos[randy]
                if x <half: x=half
                if y <half: y=half

                if x >(self.dims-half): x=self.dims-half
                if y >(self.dims-half): y=self.dims-half        
                
                # print(x-half, x+half, y-half, y+half)
                X_train[i] = img[x-half:x+half,y-half:y+half]
                Y_train[i] = mk[x-half:x+half,y-half:y+half]
        else:
            for i in range(sz):                
                x= np.random.randint(128,896)
                y= np.random.randint(128,896)                
                X_train[i] = img[x-half:x+half,y-half:y+half]               

        return X_train, Y_train


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print('idx:',idx)
        if self.train:
          
            if self.val:
                print('idx',idx)
                img, mk = self.getImage(idx)
                img, mk = self.getPatches(img, mk)
                img = np.expand_dims(img, axis=1)
                mk = np.expand_dims(mk, axis=1)
                return img, mk
            
            # print('about to sample')
            #get an index 
            idx = self.sample()
            #get image
            img, mk = self.getImage(idx)
            #get augmentations
            img, mk = self.transform_(img, mk)
            #break into patches
            img, mk = self.getPatches(img, mk)
            img = np.expand_dims(img, axis=1)
            mk = np.expand_dims(mk, axis=1)

            return img, mk
        
        # print('made it outside')
        # img = self.resize(idx)
        # img = img.reshape(1,self.dims, self.dims)
        # return img
        print('Should not be here, use the test dataset')
    

















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
        