

import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.sparse import csc_matrix, save_npz, load_npz

def dicom_to_dict(dicom_data, file_path, rles_df=None, encoded_pixels=True):
    """Parse DICOM dataset and returns a dictonary with relevant fields.

    Args:
        dicom_data (dicom): chest x-ray data in dicom format.
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): if True we will search for annotation.
        
    Returns:
        dict: contains metadata of relevant fields.
    """
    
    data = {}
    
    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID
    
    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values
       
        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1':
                pneumothorax = True
        
        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)
        
    return data


#default path is 'siim/dicom-images-train/'
def getTrainData(path_, rles_df):
    # create a list of all the files
    train_fns = sorted(glob(path_+'*/*/*.dcm'))
    # parse train DICOM dataset
    # train_metadata_df = pd.DataFrame()
    train_metadata_list = []
    for file_path in train_fns:
        dicom_data = pydicom.dcmread(file_path)
        train_metadata = dicom_to_dict(dicom_data, file_path, rles_df)
        train_metadata_list.append(train_metadata)
    return pd.DataFrame(train_metadata_list)

#default path is 'siim/dicom-images-test/'
def getTestData(path_, rles_df=None):
    # create a list of all the files
    test_fns = sorted(glob(path_+'*/*/*.dcm'))
    # parse test DICOM dataset
    # test_metadata_df = pd.DataFrame()
    test_metadata_list = []
    for file_path in test_fns:
        dicom_data = pydicom.dcmread(file_path)
        test_metadata = dicom_to_dict(dicom_data, file_path, rles_df=None, encoded_pixels=False)
        test_metadata_list.append(test_metadata)
    return pd.DataFrame(test_metadata_list)

def plot_pixel_array(data, figsize=(10,10)):
    dat = pydicom.dcmread(data['file_path'])
    plt.figure()
    plt.subplots(figsize=figsize)
    plt.imshow(dat.pixel_array, cmap=plt.cm.bone)
    if data['has_pneumothorax']:
#         mask = rle2mask(data['encoded_pixels_list'], 1024,1024)
#         plt.imshow(mask, alpha=.4)
        mask_decoded_list = [rle2mask(mask_encoded, 1024, 1024).T for mask_encoded in data['encoded_pixels_list']]
#         print('mask', mask_decoded_list.shape)
        for mask_decode in mask_decoded_list:
            plt.imshow(mask_decode, alpha=.4)
    plt.show()


def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

def saveMask(df):
    for i in range(len(df)):
        mask_ = np.zeros((1024,1024))
        mask_decoded_list = [rle2mask(mask_encoded, 1024, 1024).T for mask_encoded in df.iloc[i]['encoded_pixels_list']]
    
        for mask_decode in mask_decoded_list:
            mask_+=mask_decode
        
        mask_.astype("uint8")
        
        sprse = csc_matrix(mask_)
        save_npz('siim/mask/'+df.iloc[i]['id'], sprse)