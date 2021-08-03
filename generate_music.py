from models import Discriminator,Generator,RefinementNet
import data_loader 
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import os
from utils import img2midi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sagan = Generator.AttentionGenerator()

ref_net = RefinementNet.RefinementNet()

class Musegan(nn.Module):
    def __init__(self,gan,refinement):
        super(Musegan, self).__init__()
        self.front_gan =gan
        self.refinement= refinement
        
    def forward(self,noise):
        gan_out = self.front_gan.forward(noise)
        out = self.refinement.forward(gan_out)
        return out
    

# In[]

model = model = Musegan(sagan,ref_net)

# In[]

model.load_state_dict(torch.load('./final_self_attention_gan_model_200'))

# In[]
epoch ='Final'
fixed_noise = torch.randn(1, 100, 1, 1)
fake_song =model.forward(fixed_noise)[0]
           
fake_midi_img = fake_song.cpu().detach().numpy()
fake_midi_img =np.transpose(fake_midi_img,(1,2,0))
song_mirror = np.zeros((106,270,3))
song_mirror_03 = np.zeros((106,270,3))
for c,one in enumerate(fake_midi_img[:,:,0]):
    for h,two in enumerate(one):
        if(two>0.5):
            song_mirror[c,h,2]=255
            song_mirror[c,h,1]=255
            song_mirror[c,h,0]=255
        if(two>0.3):
            song_mirror_03[c,h,2]=255
            song_mirror_03[c,h,1]=255
            song_mirror_03[c,h,0]=255
song_mirror_03[:18:,:] = 0
song_mirror_03[75:,:,:] = 0
save_path= 'result'

song_mirror03 = Image.fromarray(np.uint8(song_mirror_03))
song_mirror = Image.fromarray(np.uint8(song_mirror))
#song_mirror03.save(os.path.join(save_path,(str(epoch)+'_0.3thresh_fake_midi_img.jpg')))
#song_mirror.save(os.path.join(save_path,(str(epoch)+'_0.5thresh_fake_midi_img.jpg')))
fake_song =torchvision.transforms.ToPILImage()((fake_song*255))

result = Image.fromarray(np.uint8(fake_song))

from utils.img2midi import image2midi

image2midi(song_mirror,'test','_')

