import os
import numpy as np
import matplotlib.pyplot as plt
#import midi2audio
import pylab
import librosa, IPython
import cv2
base_path = './data_'
save_path = './midi_img_v2'
times=1
# In[15]:
    
file_list = [ os.path.join(base_path,one) for one in (os.listdir(base_path))]




# In[15]:
from utils import midi2img
count =3333;

for one in file_list:
    if('previout_midi' in one):
        continue
    midi2img.midi2image(one,save_path,count)
    count+=1

#midi2img.midi2image(file_list[0],save_path,times)
#midi2img.midi2image(file_list[2],save_path,times)
#midi2img.midi2image(file_list[3],save_path,times)
#midi2img.midi2image(file_list[4],save_path,times)


# In[15]:
#from utils import img2midi
#img2midi.image2midi(file_list[2])

import data_loader

dloder = data_loader.DatasetLoader(save_path)



img_list = dloder.getFilePathList()
# In[15]:
w_min=0
w_max = 0
h_min=9999
h_max = 0

def find_min_max_w_max_h_min_h_max(img_list,h_max=0,h_min=9999):
    for path in img_list:
        tmp = cv2.imread(path)
        for h,one in enumerate(tmp):
            for w,two in enumerate(one):
                for c,thr in enumerate(two):
                    #print(thr)
                    if(thr== 255):
                        if( h_min > h):
                            h_min = h
                        if( h_max<h):
                            h_max = h
    return h_max,h_min

    



# In[15]:

hmax,hmin = find_min_max_w_max_h_min_h_max(img_list[:3])


www = cv2.imread(img_list[0])



# In[15]:
    
res_path = './result'

import cv2
test = cv2.imread('result/159_0.3thresh_fake_midi_img.jpg')
test[:18:,:] = 0
test[75:,:,:] = 0
test

from PIL import Image
from utils.img2midi import image2midi

ess = Image.fromarray(test)


image2midi(ess,'result','ddsdd')

# In[15]:
    
import pretty_midi
import matplotlib.pyplot as plt
import py_midicsv as pm
#ss =pretty_midi.PrettyMIDI('./holy_night.mid')

#eees =pm.midi_to_csv('./holy_night.mid')