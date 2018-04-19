import torch
import torch.nn as nn
import torch.nn.parallel

class encoder(nn.Module):
    def __init__(self, h_dim=64, n=128, size=3):
        """
        size=3, 32x32
        size=4, 64x64
        size=5, 128x128
        """
        super(encoder, self).__init__()
        self.n = n
        self.size = size
        self.linear = nn.Linear(in_features=8*8*size*n, out_features=h_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n, kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True)
        )
        self.subsampling = nn.MaxPool2d(kernel_size=2, stride=2)
        layers = []
        for i in xrange(1,size+1):
            if i == size:
                layers.append(self._make_layer(i*n, i*n))
            else:
                layers.append(self._make_layer(i*n, (i+1)*n)) 
        self.layers =  nn.ModuleList(layers) 

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
        output = self.layers[0](output)

        for i in xrange(1, self.size):
            output = self.subsampling(output)
            output = self.layers[i](output) 

        output = output.view(-1, 8 * 8 * self.size * self.n)
        output = self.linear(output)

        return output        

class decoder(nn.Module):
    def __init__(self, h_dim=64, n=128, size=3):
        super(decoder, self).__init__()
        self.n = n
        self.size = size
        self.linear = nn.Linear(in_features=64, out_features=8*8*n)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        upsamplings = []
        for i in xrange(1, size):
            upsamplings.append(nn.Upsample(scale_factor=2**i, mode='nearest'))
        self.upsamplings = nn.ModuleList(upsamplings)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=3,kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True)
        )
        layers = []
        for i in xrange(1, self.size+1):
            if i==1:
                layers.append(self._make_layer(in_dim=n, out_dim=n))
            else:
                layers.append(self._make_layer(in_dim=2*n, out_dim=n)) 
        self.layers = nn.ModuleList(layers)

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

        output = self.layers[0](h0)

        for i in xrange(1, self.size):
            output = self.upsampling(output)
            residual = self.upsamplings[i-1](h0)
            output = torch.cat((output, residual), 1)
            output = self.layers[i](output)

        output = self.conv(output)
        return output 

class BEGAN_G(nn.Module):
    def __init__(self,size=3):
        """
        size=3, 32x32
        size=4, 64x64
        size=5, 128x128
        """
        super(BEGAN_G, self).__init__()
        self.decoder = decoder(size=size)

    def forward(self, input):
        output = self.decoder(input)
        return output


class BEGAN_D(nn.Module):
    def __init__(self, size=3):
        """
        size=3, 32x32
        size=4, 64x64
        size=5, 128x128
        """
        super(BEGAN_D, self).__init__()
        self.encoder = encoder(size=size)
        self.decoder = decoder(size=size)
        
    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output
