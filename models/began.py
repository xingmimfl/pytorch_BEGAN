import torch
import torch.nn as nn
import torch.nn.parallel

class encoder(nn.Module):
    def __init__(self, h_dim=64, n=128):
        super(encoder, self).__init__()
        self.n = n
        self.linear = nn.Linear(in_features=8*8*3*n, out_features=h_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n, kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True)
        )
        self.subsampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(n, 2*n)
        self.layer2 = self._make_layer(2*n, 3*n)
        self.layer3 = self._make_layer(3*n, 3*n)
         
    def _make_layer(self,in_dim, out_dim):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True),
        )
        return layers

    def forward(self, input):
        output = self.conv(input)
        output = self.layer1(output)

        output = self.subsampling(output)
        output = self.layer2(output) 

        output = self.subsampling(output)
        output = self.layer3(output)
        
        output = output.view(-1, 8 * 8 * 3 * self.n)
        output = self.linear(output)

        return output        

class decoder(nn.Module):
    def __init__(self, h_dim=64, n=128):
        super(decoder, self).__init__()
        self.n = n
        self.linear = nn.Linear(in_features=64, out_features=8*8*n)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsampling1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsampling2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=3,kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True)
        )
        self.layer1 = self._make_layer(in_dim=n, out_dim=n)
        self.layer2 = self._make_layer(in_dim=2*n, out_dim=n)
        self.layer3 = self._make_layer(in_dim=2*n, out_dim=n)
        #self.layer4 = _make_layer(skip_con=True) #--64x64
        #self.layer5 = _make_layer(skip_con=True) #--128x128
    def _make_layer(self,in_dim, out_dim):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True),
        )
        return layers
         
    def forward(self, input):
        h0 = self.linear(input)
        h0 = h0.view(-1,self.n,8,8)

        output = self.layer1(h0)

        output = self.upsampling(output)
        residual = self.upsampling1(h0)
        output = torch.cat((output, residual), 1)
        output = self.layer2(output)

        output = self.upsampling(output)
        residual = self.upsampling2(h0)
        output = torch.cat((output, residual), 1)
        output = self.layer3(output)

        output = self.conv(output)
        return output 

class BEGAN_G(nn.Module):
    def __init__(self):
        super(BEGAN_G, self).__init__()
        self.decoder = decoder()

    def forward(self, input):
        output = self.decoder(input)
        return output


class BEGAN_D(nn.Module):
    def __init__(self):
        super(BEGAN_D, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        
    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output
