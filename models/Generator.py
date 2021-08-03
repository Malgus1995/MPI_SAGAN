import torch 
import torch.nn as nn
import torch.optim as optim
ngf = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
# In[] 

class VAE_GAN(nn.Module):
    def __init__(self):
        super(VAE_GAN, self).__init__()
        self.lstm = nn.LSTM(100, 128, 4, batch_first=True)

                
# In[]


        
# In[] 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, ngf * 16, 4, (1,2), 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 3, (1,2), 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 8, ngf * 4, (4,2), (1,2), 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf*2, (4,1), (3,4), 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 3, (2,2), (2,1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 1, 4, 2, (1,3), bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
# In[]

g =  Generator().to(device)


noise = torch.randn((8,100,1,1)).to(device)


eee =g.forward(noise)

print(eee.shape)
    
# In[]
ngf=64
class CycleGenerator(nn.Module):
    def __init__(self):
        super(CycleGenerator, self).__init__()
        self.convt1= nn.ConvTranspose2d( 100, 256, 128, 6,1,4, bias=False)
        self.convd_1 = nn.Conv2d(256, 64, 7,1)
        self.convd_2 = nn.Conv2d(64, 128, 3,2)
        self.convd_3 = nn.Conv2d(128, 256, 3,2)
        self.residual_convd = nn.Conv2d(256, 256, 3,1,1)
        # input is Z, going into a convolution
        self.convt2= nn.ConvTranspose2d(256, 256, 3, (2,5), 0, bias=False) 

        self.convt3= nn.ConvTranspose2d( 256, 128, 3, 2, bias=False)
        #nn.BatchNorm2d(ngf*2),
        #nn.ReLU(True),
        self.convt4= nn.ConvTranspose2d( 128, 64, 3,1,(4,8), bias=False)

        self.convt5= nn.ConvTranspose2d( 64, 32, 4,1,(2,4), bias=False)
        self.convt6= nn.ConvTranspose2d( 32, 3, 7,1,(6,5), bias=False)
        self.finalconv= nn.Conv2d( 3, 1, 7,1,1, bias=False)
        
        self.batchnorm256 =nn.BatchNorm2d(256)
        self.batchnorm128 =nn.BatchNorm2d(128)
        self.batchnorm64 =nn.BatchNorm2d(64)
        self.batchnorm32 =nn.BatchNorm2d(32)
        self.batchnorm3 =nn.BatchNorm2d(3)
        #nn.Tanh()
            # state size. (nc) x 64 x 64

    def forward(self, x):
        out = self.convt1(x)
        out = self.batchnorm256(out)
        out = self.convd_1(out)
        out =self.batchnorm64(out)
        out = nn.LeakyReLU()(out)
        out = self.convd_2(out)
        out = self.batchnorm128(out)
        out = nn.LeakyReLU()(out)
        out = self.convd_3(out)
        out = self.batchnorm256(out)
        out = nn.LeakyReLU()(out)
        
        residual_out_1 =self.residual_convd(out)+out
        residual_out_1 = self.batchnorm256(residual_out_1)
        residual_out_1 = nn.LeakyReLU()(residual_out_1)
        
        residual_out_2 = self.residual_convd(residual_out_1)+residual_out_1
        residual_out_2 = self.batchnorm256(residual_out_2)
        residual_out_2 = nn.LeakyReLU()(residual_out_2)
        
        residual_out_3 = self.residual_convd(residual_out_2)+residual_out_2
        residual_out_3 = self.batchnorm256(residual_out_3)
        residual_out_3 = nn.LeakyReLU()(residual_out_3)
        
        residual_out_4 = self.residual_convd(residual_out_3)+residual_out_3
        residual_out_4 = self.batchnorm256(residual_out_4)
        residual_out_4 = nn.LeakyReLU()(residual_out_4)
        
        residual_out_5 = self.residual_convd(residual_out_4)+residual_out_4
        residual_out_5 = self.batchnorm256(residual_out_5)
        residual_out_5 = nn.LeakyReLU()(residual_out_5)
        
        residual_out_6 = self.residual_convd(residual_out_5)+residual_out_5
        residual_out_6 = self.batchnorm256(residual_out_6)
        residual_out_6 = nn.LeakyReLU()(residual_out_6)
        

        transout = self.convt2(residual_out_6)
        transout = self.batchnorm256(transout)
        transout = nn.LeakyReLU()(transout)
        
        transout = self.convt3(transout)
        transout = self.batchnorm128(transout)
        transout = nn.LeakyReLU()(transout)
        
        
        transout = self.convt4(transout)
        transout = self.batchnorm64(transout)
        transout = nn.LeakyReLU()(transout)
        
        transout = self.convt5(transout)
        transout = self.batchnorm32(transout)
        transout = nn.LeakyReLU()(transout)
        
        transout = self.convt6(transout)
        transout = self.batchnorm3(transout)
        transout = nn.LeakyReLU()(transout)
        final = self.finalconv(transout)
        final = nn.Tanh()(final)
        return final
        
    
        
# In[]
#next(cg.parameters()).device


cg =  CycleGenerator().to(device)


noise = torch.randn((8,100,1,1)).to(device)


eee =cg.forward(noise)

print(eee.shape)
        
# In[]
ngf=64
class AttentionGenerator(nn.Module):
    def __init__(self):
        super(AttentionGenerator, self).__init__()
        self.convt1= nn.ConvTranspose2d( 100, 256, 32, 2,1,0, bias=False)

        self.conv1x1d = nn.Conv2d(256, 256, 1,bias=False)
        self.convt = nn.ConvTranspose2d(256, 256 , 1,bias=False)
        # input is Z, going into a convolution
        self.convt2= nn.ConvTranspose2d(256, 256, 3, (2,5), 0, bias=False) 

        self.convt3= nn.ConvTranspose2d( 256, 128, 3, 2, bias=False)
        #nn.BatchNorm2d(ngf*2),
        #nn.ReLU(True),
        self.convt4= nn.ConvTranspose2d( 128, 64, 3,1,(4,8), bias=False)

        self.convt5= nn.ConvTranspose2d( 64, 32, 4,1,(2,4), bias=False)
        self.convt6= nn.ConvTranspose2d( 32, 3, 7,1,(6,5), bias=False)
        self.finalconv= nn.Conv2d( 3, 1, 7,1,1, bias=False)
        
        self.batchnorm256 =nn.BatchNorm2d(256)
        self.batchnorm128 =nn.BatchNorm2d(128)
        self.batchnorm64 =nn.BatchNorm2d(64)
        self.batchnorm32 =nn.BatchNorm2d(32)
        self.batchnorm3 =nn.BatchNorm2d(3)
        #nn.Tanh()
            # state size. (nc) x 64 x 64
            
    def selfattention_module(self,combined_layer):
        
        conv_out1x1 = self.conv1x1d(combined_layer)
        conv_out1x1 = self.batchnorm256(conv_out1x1)  
        conv_out1x1 = nn.LeakyReLU()(conv_out1x1)
        conv_out1x1 = self.convt(conv_out1x1)
        
        conv_out1x1_2 = self.conv1x1d(combined_layer)
        conv_out1x1_2 = self.batchnorm256(conv_out1x1_2)  
        conv_out1x1_2 = self.convt(conv_out1x1)
        
        attenionmap = nn.Softmax()(torch.matmul(conv_out1x1,conv_out1x1_2))
        output = torch.matmul(attenionmap, combined_layer).contiguous()
        return output
        

    def forward(self, x):
        conv1_out = self.convt1(x)
        output = self.selfattention_module(conv1_out)
        output = self.convt2(output)
        output = self.batchnorm256(output)
        output = nn.LeakyReLU()(output)
        
        output = self.convt3(output)
        output = self.batchnorm128(output)
        output = nn.LeakyReLU()(output)
        
        output = self.convt4(output)
        output = self.batchnorm64(output)
        output = nn.LeakyReLU()(output)
        
        output = self.convt5(output)
        output = self.batchnorm32(output)
        output = nn.LeakyReLU()(output)
        
        output = self.convt6(output)
        output = self.batchnorm3(output)
        output = nn.LeakyReLU()(output)
        
        output = self.finalconv(output)
        output = nn.Tanh()(output)
        return output

    def get_conv_feature(self,x):
        conv1_out = self.convt1(x)
        output = self.selfattention_module(conv1_out)
        output = self.convt2(output)
        output = self.batchnorm256(output)
        output = nn.LeakyReLU()(output)

        output = self.convt3(output)
        output = self.batchnorm128(output)
        output = nn.LeakyReLU()(output)

        output = self.convt4(output)
        output = self.batchnorm64(output)
        output = nn.LeakyReLU()(output)

        output = self.convt5(output)
        output = self.batchnorm32(output)
        output = nn.LeakyReLU()(output)

        output = self.convt6(output)
        output = self.batchnorm3(output)
        output = nn.LeakyReLU()(output)

        output = self.finalconv(output)
        return output


        
        
# In[] 

ag = AttentionGenerator()
noise = torch.randn((8,100,1,1))

print(ag.forward(noise).shape)




