import torch
import torch.nn as nn





class discriminator_model(nn.Module):
    def __init__(self,input_channels=6,out_channel_1 = 64, kernel_size = 4, stride = 2, padding = 1, num_blocks=4):
        super(discriminator_model, self).__init__()
        
        self.initial = nn.Sequential(nn.Conv2d(input_channels, out_channel_1, kernel_size= kernel_size, stride = stride, padding = padding, padding_mode = 'reflect',bias=False),
                                    nn.LeakyReLU(0.2))
        self.inter_block = nn.Sequential(*[self.block(out_channel_1*i, out_channel_1*2*i, kernel_size, stride = 1 if i==num_blocks else stride, padding=padding) for i in range(1,num_blocks+1) if i != 3])
        self.out = nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 1, bias=False)
        self.act = nn.Sigmoid()
        self.disc_module = nn.Sequential(*[self.initial,self.inter_block,self.out,self.act])
        self.apply(self.__init__weights)
        
        
    def __init__weights(self, module):
        
        if isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight.data, mean = 0.0, std= 0.02)
           
            
        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight.data, mean = 0.0, std= 0.02)  
            
    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        
        
               
        return nn.Sequential(nn.Conv2d(in_channels , out_channels, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect',bias=False),
                     nn.BatchNorm2d(out_channels),nn.LeakyReLU(0.2))
    
    
    def forward(self, x):
        
        
        return self.disc_module(x)
    



