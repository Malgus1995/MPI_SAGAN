from models import Discriminator, Generator, RefinementNet
import data_loader
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import os
from utils import img2midi

img_base_path = './prevmidi_img'
batch_size = 4

Dataset = data_loader.DatasetLoader(img_base_path, batch_size)

lr = 0.0001
beta1 = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# In[]
gan = Generator.Generator()
sagan = Generator.AttentionGenerator()
discriminator = Discriminator.Discriminator()
ref_net = RefinementNet.RefinementNet()


#
class Musegan(nn.Module):
    def __init__(self, gan, refinement):
        super(Musegan, self).__init__()
        self.front_gan = gan
        self.refinement = refinement

    def forward(self, noise):
        gan_out = self.front_gan.forward(noise)
        out = self.refinement.forward(gan_out)
        return out

    def get_attention_feature(self, noise):
        feature_out = self.front_gan.get_conv_feature(noise)
        return feature_out


musegan = Musegan(sagan, ref_net)

musegan = musegan.to(device)

discriminator = discriminator.to(device)
# In[]
criterion = nn.BCELoss()

real_label = 1
fake_label = 0
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(musegan.parameters(), lr=lr, betas=(beta1, 0.999))
fixed_noise = torch.randn(1, 100, 1, 1).to(device)
num_epochs = 501

# In[]
d_losses, g_losses, real_scores, fake_scores = [], [], [], []


def train(discriminator, generator, dataloader, num_epochs, optimizerG):
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            data = data.to(device)
            data = data.type(torch.cuda.FloatTensor)
            discriminator.zero_grad()

            real_labels = torch.ones(len(data), 1).to(device)
            output = discriminator.forward(data)
            rg_loss = criterion(output, real_labels)
            rg_loss.backward()

            fake_labels = torch.zeros(len(data), 1).to(device)
            noise = torch.randn(len(data), 100, 1, 1).to(device)
            fake = generator.forward(noise)

            output = discriminator.forward(fake)
            fd_loss = criterion(output, fake_labels)
            fd_loss.backward()
            d_loss = fd_loss + rg_loss
            d_losses.append(d_loss)
            optimizerD.step()

            generator.zero_grad()
            fake = generator.forward(noise)
            # thresh =nn.threshold(0.3,1)
            # fake = thresh(fake)
            g_loss = criterion(discriminator.forward(fake), real_labels)

            g_loss.backward()
            optimizerG.step()
            if i % 50 == 0:
                print('Epoch :[%d/%d],\t G_loss: %.4f,\t D_loss: %.4f' % (epoch + 1, num_epochs, g_loss, d_loss))
            if ((epoch % 10 == 0) or g_loss < 1 or d_loss >1):
                torch.save(generator.state_dict(), './self_attention_gan_model_' + str(epoch))
                print(epoch, ",", 'fake_song_saved')
                fake_song = generator.forward(fixed_noise)[0]

                fake_midi_img = fake_song.cpu().detach().numpy()
                fake_midi_img = np.transpose(fake_midi_img, (1, 2, 0))
                song_mirror = np.zeros((106, 270, 3))
                song_mirror_03 = np.zeros((106, 270, 3))
                song_mirror_07 = np.zeros((106, 270, 3))
                song_mirror_007 = np.zeros((106, 270, 3))
                for c, one in enumerate(fake_midi_img[:, :, 0]):
                    for h, two in enumerate(one):
                        if (two > 0.7):
                            song_mirror_07[c, h, 2] = 255
                            song_mirror_07[c, h, 1] = 255
                            song_mirror_07[c, h, 0] = 255

                            song_mirror_007[c, h, 2] = 255
                            song_mirror_007[c, h, 1] = 255
                            song_mirror_007[c, h, 0] = 255
                        if (two > 0.5):
                            song_mirror[c, h, 2] = 255
                            song_mirror[c, h, 1] = 255
                            song_mirror[c, h, 0] = 255
                        if (two > 0.3):
                            song_mirror_03[c, h, 2] = 255
                            song_mirror_03[c, h, 1] = 255
                            song_mirror_03[c, h, 0] = 255

                song_mirror_03[:18:, :] = 0
                song_mirror_03[75:, :, :] = 0
                song_mirror_007[:18:, :] = 0
                song_mirror_007[75:, :, :] = 0
                save_path = 'result'
                song_mirror03 = Image.fromarray(np.uint8(song_mirror_03))
                song_mirror = Image.fromarray(np.uint8(song_mirror))
                song_mirror_07 = Image.fromarray(np.uint8(song_mirror_07))
                song_mirror_007 = Image.fromarray(np.uint8(song_mirror_07))
                song_mirror_007 = Image.fromarray(np.uint8(song_mirror_007))
                song_mirror_007.save(os.path.join(save_path, (str(epoch) + '_0.7v2thresh_fake_midi_img.jpg')))
                song_mirror_07.save(os.path.join(save_path, (str(epoch) + '_0.7thresh_fake_midi_img.jpg')))
                song_mirror03.save(os.path.join(save_path, (str(epoch) + '_0.3thresh_fake_midi_img.jpg')))
                song_mirror.save(os.path.join(save_path, (str(epoch) + '_0.5thresh_fake_midi_img.jpg')))
                fake_song = torchvision.transforms.ToPILImage()((fake_song * 255))

                fake_song.save(os.path.join(save_path, (str(epoch) + 'fake_midi_img.jpg')))
                # img2midi.image2midi(fake_song,save_path,('fake_song'+str(epoch)))
        g_losses.append(g_loss)
        d_losses.append(d_loss)

    return generator


# In[]
# gen = train(discriminator, musegan, Dataset.dataload(), num_epochs,optimizerG)

# In[]


# In[]
# gan_test_result = musegan.forward(fixed_noise)

# In[]

from PIL import Image
import numpy as np
import cv2

# test = cv2.imread('result/170_0.3thresh_fake_midi_img.jpg')

# ess = Image.fromarray(np.uint8(test))

# from utils.img2midi import image2midi

# image2midi(ess,'result','ddsdd')

# In[]

# test_y = gan_test_result.cpu().detach().numpy()
# numpy_res_y =np.transpose(test_y[0],(1,2,0))

# numpy_res_y_mirror = np.zeros(numpy_res_y.shape)
# thres holding
# for ch in range(0,3):
#    for c,one in enumerate(numpy_res_y_mirror[:,:,ch]):
#        for h,two in enumerate(one):
#            if(two>0.5):
#                numpy_res_y_mirror[c,h,ch]=255


# In[]

# ess = Image.fromarray(np.uint8(numpy_res_y_mirror))


# image2midi(ess,'result',' ')


# In[]
# cmusegan = Musegan(cygan, ref_net)
# cmusegan= cmusegan.to(device)
# optimizerG = optim.Adam(cmusegan.parameters(), lr=lr, betas=(beta1, 0.999))

# cgen = train(discriminator, cmusegan, Dataset.dataload(), num_epochs,optimizerG)


# In[]
smusegan = Musegan(sagan, ref_net)
smusegan = smusegan.to(device)
optimizerG = optim.Adam(smusegan.parameters(), lr=lr, betas=(beta1, 0.999))

# In[]
sgen = train(discriminator, smusegan, Dataset.dataload(), num_epochs, optimizerG)

# In[]

# model.load_state_dict(torch.load('/final_self_attention_gan_model_200'))


# In[]
import matplotlib.pyplot as plt
import seaborn

g_indexes = [i for i in range(0, len(g_losses))]
plt.plot(g_indexes[0:300], g_losses[0:300], color='red')
plt.plot(g_indexes[0:300], d_losses[0:300], color='blue')
plt.xlabel('epochs')
plt.ylabel('Self-Attention Gan and D losses')
plt.legend(['red=d_loss', 'blue=g_loss'])

