import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import Generator,RefinementNet
#import RefinementNet,Generator
ngf = 64
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
    def get_attention_feature(self,noise):
        feature_out = self.front_gan.get_conv_feature(noise)
        return feature_out

#pretrained_musegan = Musegan(sagan,ref_net)
#pretrained_musegan.load_state_dict(torch.load('./self_attention_gan_model_130'))
class SelfAttentionUnit(nn.Module):
    def __init__(self,pretrained_unit_path):
        super(SelfAttentionUnit, self).__init__()
        self.pretrained_musegan = Musegan(sagan,ref_net)
        self.pretrained_musegan.load_state_dict(torch.load(pretrained_unit_path))
        self.pretrained_musegan = Musegan(sagan, ref_net).front_gan


        for param in self.pretrained_musegan.parameters():
            param.requires_grad = False

    def get_conv_feature(self,x):
        output = self.pretrained_musegan.get_conv_feature(x)
        return output




class StackedAttentionGenerator(nn.Module):
    def __init__(self,model_path):
        super(StackedAttentionGenerator, self).__init__()
        self.attention_gan_unit = SelfAttentionUnit(model_path)

        self.lrelu = F.leaky_relu
        self.incep_conv2d_1x1_16 = nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_1x1 = nn.Conv2d(1,2,kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_1x1_8 = nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_1x1_8_32 = nn.Conv2d(8, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_1x1_32 = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_3x3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.incep_conv2d_3x3_8_32 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.incep_conv2d_3x3_32 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.incep_dropout = nn.Dropout(0.2)

        self.conv1x1_fin = nn.Conv2d(3, 1, 1,stride=1, padding=0, bias=False)
        self.conv1x1d = nn.Conv2d(256, 256, 1, bias=False)
        self.convt = nn.ConvTranspose2d(256, 256, 1, bias=False)


        self.aconv2d = nn.Conv2d(256,256,(1,10),(1,2),(2,0),bias=False)
        self.aconv2d_2 = nn.Conv2d(256, 256, (1,10),(2,2),(6,0), bias=False)

        self.aconvt2d = nn.ConvTranspose2d(256, 256, (3,7), (2,5), 0,bias=False)
        self.aconv2d_3 = nn.ConvTranspose2d(256, 256, 3,1,0,bias=False)

        self.conv3x3_64 = nn.Conv2d(128, 64,kernel_size=3, stride=1, padding=1, bias=True)
        self.conv64_16 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.finalconv = nn.Conv2d(16, 1, 3, 1, 1, bias=False)

        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm16 = nn.BatchNorm2d(16)
        self.batchnorm8 = nn.BatchNorm2d(8)
        self.batchnorm2 = nn.BatchNorm2d(2)
        self.batchnorm4 = nn.BatchNorm2d(4)
        self.batchnorm3 = nn.BatchNorm2d(3)
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.tanh =nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # state size. (nc) x 64 x 64

    def inception_m(self,feature_input):
        inception_line1 = self.lrelu(self.incep_conv2d_1x1_8(feature_input))
        inception_line1 = self.lrelu(self.incep_conv2d_3x3(inception_line1))
        inception_line1 = self.batchnorm16(inception_line1)
        inception_line1 = self.lrelu(self.incep_conv2d_3x3_32(inception_line1))

        inception_line2 = self.lrelu(self.incep_conv2d_1x1_8(feature_input))
        inception_line2 = self.batchnorm8(inception_line2)
        inception_line2 = self.lrelu(self.incep_conv2d_1x1_8_32(inception_line2))

        inception_line3 = self.lrelu(self.incep_dropout(feature_input))
        inception_line3 = self.lrelu(self.incep_conv2d_1x1_8(inception_line3))
        inception_line3 = self.batchnorm8(inception_line3)
        inception_line3 = self.lrelu(self.incep_conv2d_1x1_8_32(inception_line3))

        inception_line4 = self.lrelu(self.incep_conv2d_1x1_32(feature_input))


        return torch.cat([inception_line1, inception_line2,inception_line3,inception_line4],1)

    def forward_using_fixed_noise(self,fixed_noise):

        pt_gan_incep1 = self.inception_m(self.attention_gan_unit.get_conv_feature(fixed_noise))
        pt_gan_incep2 = self.inception_m(self.attention_gan_unit.get_conv_feature(fixed_noise))
        pt_gan_incep3 = self.inception_m(self.attention_gan_unit.get_conv_feature(fixed_noise))
        pt_gan_incep4 = self.inception_m(self.attention_gan_unit.get_conv_feature(fixed_noise))

        concatten_inception = torch.cat([pt_gan_incep1,pt_gan_incep2,pt_gan_incep3, pt_gan_incep4],3)

        output = self.conv3x3_64(concatten_inception)
        output = self.batchnorm64(output)
        output = self.conv64_16(output)
        output = self.lrelu(output)
        output = self.finalconv(output)
        output = self.sigmoid(output)

        return output

    def forward_using_diff_noise(self,diff_noise_1,diff_noise_2,diff_noise_3,diff_noise_4):
        pt_gan_incep1 = self.inception_m(self.attention_gan_unit.get_conv_feature(diff_noise_1))
        pt_gan_incep2 = self.inception_m(self.attention_gan_unit.get_conv_feature(diff_noise_2))
        pt_gan_incep3 = self.inception_m(self.attention_gan_unit.get_conv_feature(diff_noise_3))
        pt_gan_incep4 = self.inception_m(self.attention_gan_unit.get_conv_feature(diff_noise_4))

        concatten_inception = torch.cat([pt_gan_incep1,pt_gan_incep2,pt_gan_incep3, pt_gan_incep4],3)

        output = self.conv3x3_64(concatten_inception)
        output = self.batchnorm64(output)
        output = self.conv64_16(output)
        output = self.lrelu(output)
        output = self.finalconv(output)
        output = self.sigmoid(output)
        return output


# In[]

class StackedAttentionGenerator_m_diff(nn.Module):
    def __init__(self,model_path1,model_path2,model_path3,model_path4):
        super(StackedAttentionGenerator_m_diff, self).__init__()
        self.attention_gan_unit_1 = SelfAttentionUnit(model_path1)
        self.attention_gan_unit_2 = SelfAttentionUnit(model_path2)
        self.attention_gan_unit_3 = SelfAttentionUnit(model_path3)
        self.attention_gan_unit_4 = SelfAttentionUnit(model_path4)

        self.lrelu = F.leaky_relu
        self.incep_conv2d_1x1_16 = nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_1x1 = nn.Conv2d(1,2,kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_1x1_8 = nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_1x1_8_32 = nn.Conv2d(8, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_1x1_32 = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.incep_conv2d_3x3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.incep_conv2d_3x3_8_32 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.incep_conv2d_3x3_32 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.incep_dropout = nn.Dropout(0.2)

        self.conv1x1_fin = nn.Conv2d(3, 1, 1,stride=1, padding=0, bias=False)
        self.conv1x1d = nn.Conv2d(256, 256, 1, bias=False)
        self.convt = nn.ConvTranspose2d(256, 256, 1, bias=False)


        self.aconv2d = nn.Conv2d(256,256,(1,10),(1,2),(2,0),bias=False)
        self.aconv2d_2 = nn.Conv2d(256, 256, (1,10),(2,2),(6,0), bias=False)

        self.aconvt2d = nn.ConvTranspose2d(256, 256, (3,7), (2,5), 0,bias=False)
        self.aconv2d_3 = nn.ConvTranspose2d(256, 256, 3,1,0,bias=False)

        self.conv3x3_64 = nn.Conv2d(128, 64,kernel_size=3, stride=1, padding=1, bias=True)
        self.conv64_16 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.finalconv = nn.Conv2d(16, 1, 3, 1, 1, bias=False)

        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm16 = nn.BatchNorm2d(16)
        self.batchnorm8 = nn.BatchNorm2d(8)
        self.batchnorm2 = nn.BatchNorm2d(2)
        self.batchnorm4 = nn.BatchNorm2d(4)
        self.batchnorm3 = nn.BatchNorm2d(3)
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.tanh =nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # state size. (nc) x 64 x 64

    def inception_m(self,feature_input):
        feature_input = self.tanh(feature_input)
        inception_line1 = self.lrelu(self.incep_conv2d_1x1_8(feature_input))
        inception_line1 = self.lrelu(self.incep_conv2d_3x3(inception_line1))
        inception_line1 = self.batchnorm16(inception_line1)
        inception_line1 = self.lrelu(self.incep_conv2d_3x3_32(inception_line1))

        inception_line2 = self.lrelu(self.incep_conv2d_1x1_8(feature_input))
        inception_line2 = self.batchnorm8(inception_line2)
        inception_line2 = self.lrelu(self.incep_conv2d_1x1_8_32(inception_line2))

        inception_line3 = self.lrelu(self.incep_dropout(feature_input))
        inception_line3 = self.lrelu(self.incep_conv2d_1x1_8(inception_line3))
        inception_line3 = self.batchnorm8(inception_line3)
        inception_line3 = self.lrelu(self.incep_conv2d_1x1_8_32(inception_line3))

        inception_line4 = self.lrelu(self.incep_conv2d_1x1_32(feature_input))


        return torch.cat([inception_line1, inception_line2,inception_line3,inception_line4],1)

    def forward_using_fixed_noise(self,fixed_noise):

        pt_gan_incep1 = self.inception_m(self.attention_gan_unit_1.get_conv_feature(fixed_noise))
        pt_gan_incep2 = self.inception_m(self.attention_gan_unit_2.get_conv_feature(fixed_noise))
        pt_gan_incep3 = self.inception_m(self.attention_gan_unit_3.get_conv_feature(fixed_noise))
        pt_gan_incep4 = self.inception_m(self.attention_gan_unit_4.get_conv_feature(fixed_noise))

        concatten_inception = torch.cat([pt_gan_incep1,pt_gan_incep2,pt_gan_incep3, pt_gan_incep4],3)

        output = self.conv3x3_64(concatten_inception)
        output = self.batchnorm64(output)
        output = self.conv64_16(output)
        output = self.lrelu(output)
        output = self.finalconv(output)
        output = self.sigmoid(output)

        return output

    def forward_using_diff_noise(self,diff_noise_1,diff_noise_2,diff_noise_3,diff_noise_4):
        pt_gan_incep1 = self.inception_m(self.attention_gan_unit_1.get_conv_feature(diff_noise_1))
        pt_gan_incep2 = self.inception_m(self.attention_gan_unit_2.get_conv_feature(diff_noise_2))
        pt_gan_incep3 = self.inception_m(self.attention_gan_unit_3.get_conv_feature(diff_noise_3))
        pt_gan_incep4 = self.inception_m(self.attention_gan_unit_4.get_conv_feature(diff_noise_4))

        concatten_inception = torch.cat([pt_gan_incep1,pt_gan_incep2,pt_gan_incep3, pt_gan_incep4],3)

        output = self.conv3x3_64(concatten_inception)
        output = self.batchnorm64(output)
        output = self.conv64_16(output)
        output = self.lrelu(output)
        output = self.finalconv(output)
        output = self.sigmoid(output)
        return output
