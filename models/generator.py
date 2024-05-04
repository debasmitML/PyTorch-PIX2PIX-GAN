import torch
import torch.nn as nn

class generator_model(nn.Module):
    def __init__(self, input_channels=3, out_channel_1 = 64, kernel_size = 4, stride = 2, padding = 1, num_blocks=4):
        super(generator_model, self).__init__()
        
        self.apply(self.__init__weights)
        
        self.initial = nn.Sequential(nn.Conv2d(input_channels, out_channel_1, kernel_size= kernel_size, stride = stride, padding = padding, padding_mode = 'reflect',bias=False),
                                    nn.LeakyReLU(0.2))
        
        self.encode1 = self.encoder_block(64, 128, 4, 2, 1)                  
        self.encode2 = self.encoder_block(128,256,4,2,1)
        self.encode3 = self.encoder_block(256,512,4,2,1)
        self.encode4 = self.encoder_block(512,512,4,2,1)
        self.encode5 = self.encoder_block(512,512,4,2,1)
        self.encode6 = self.encoder_block(512,512,4,2,1)
        
        self.bottleneck_conv = nn.Conv2d(512 , 512, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bottleneck_act = nn.ReLU()
        
        self.decode1 = self.decoder_block(512,512,4,2,1,dropout=True)
        self.decode2 = self.decoder_block(1024,512,4,2,1,dropout=True) 
        self.decode3 = self.decoder_block(1024,512,4,2,1,dropout=True)
        self.decode4 = self.decoder_block(1024,512,4,2,1,dropout=False)
        self.decode5 = self.decoder_block(1024,256,4,2,1,dropout=False)
        self.decode6 = self.decoder_block(512,128,4,2,1,dropout=False)
        self.decode7 = self.decoder_block(256,64,4,2,1,dropout=False) 
        
        
        
        self.out = nn.ConvTranspose2d(out_channel_1, input_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        self.act = nn.Tanh()
        
    def __init__weights(self, module):
        
        if isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight.data, mean = 0.0, std= 0.02)
         
        elif isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.normal_(module.weight.data, mean = 0.0, std= 0.02)   
            
        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight.data, mean = 0.0, std= 0.02)  
             
        
    def encoder_block(self,in_channels,out_channels,kernel_size,stride,padding,batch_norm=True):
        
        
        return nn.Sequential(nn.Conv2d(in_channels , out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2))
        
        
            
    def decoder_block(self, in_chan, out_chan, kernel_size, stride, padding, dropout = False):
        
        decode_list = nn.ModuleList([
            nn.ConvTranspose2d(in_chan, out_chan, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        ])
        
        if dropout:
            decode_list.insert(2,nn.Dropout(0.5))
        
        return nn.Sequential(*decode_list)
        
        
    
    def forward(self, x):
        
        e1 = self.initial(x)    ##(batch_size,64,128,128)
        e2 = self.encode1(e1)   ##(batch_size,128,64,64)
        e3 = self.encode2(e2)   ##(batch_size,256,32,32)
        e4 = self.encode3(e3)                        ##(batch_size,512,16,16)
        e5 = self.encode4(e4)                        ##(batch_size,512,8,8)
        e6 = self.encode5(e5)                        ##(batch_size,512,4,4)
        e7 = self.encode6(e6)                        ##(batch_size,512,2,2)
        
        
        bottleneck_conv = self.bottleneck_conv(e7) ##(batch_size,512,1,1)
        bottleneck_act = self.bottleneck_act(bottleneck_conv)                       ##(batch_size,512,1,1)
        
        
        d1 = self.decode1(bottleneck_act)                ##(batch_size,512,2,2)
        d2 = self.decode2(torch.cat((d1,e7),dim=1))    ##(batch_size,512,4,4)
        d3 = self.decode3(torch.cat((d2,e6),dim=1))    ##(batch_size,512,8,8)
        d4 = self.decode4(torch.cat((d3,e5),dim=1))   ##(batch_size,512,16,16)
        d5 = self.decode5(torch.cat((d4,e4),dim=1))   ##(batch_size,256,32,32)
        d6 = self.decode6(torch.cat((d5,e3),dim=1))    ##(batch_size,128,64,64)
        d7 = self.decode7(torch.cat((d6,e2),dim=1))    ##(batch_size,64,128,128)
        
        
        out = self.out(d7)
        out = self.act(out)
        
        return out
        
       
