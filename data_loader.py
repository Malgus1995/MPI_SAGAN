import os
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
import matplotlib.pyplot as plt
import pylab
import librosa, IPython
import cv2




# In[]:

class DatasetLoader():
    def __init__(self, path,batch_size=8):
        self.path = path
        self.batch_size = batch_size
        
    def getFilePathList(self):
        file_list =  os.listdir(self.path)
        
        file_path_list = [os.path.join(self.path,one) for one in file_list]
        
        return file_path_list
    
    def dataload(self):
        file_list = self.getFilePathList()
        data_length = len(file_list)
        train_data = np.zeros((data_length,1,106, 1080))
        print('data loading......')
        for i,img in enumerate(file_list):
            #print(file_list[i])
            train_data[i,:,:,:] = np.transpose(cv2.imread(img)[:,:,:1],(2,0,1)).astype('double')
        train_data /=255
        return torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,shuffle=True)
        
        
        
        
# In[]


#dloder = DatasetLoader('./midi_img')

#jebal = dloder.getFilePathList()

#data = dloder.dataload()


# In[]
# In[]
# In[]
