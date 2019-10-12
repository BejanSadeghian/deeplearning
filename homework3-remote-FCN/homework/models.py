import torch
import torch.nn.functional as F
from torch import nn

class CNNClassifier(torch.nn.Module):
    
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(n_output), #Swapped
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, stride=1, bias=False),
                    torch.nn.BatchNorm2d(n_output),
                    torch.nn.ReLU()
                    )
            self.downsample = None
            if n_input != n_output or stride != 1:
                self.downsample = torch.nn.Sequential(
                        torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                        torch.nn.BatchNorm2d(n_output)
                        )

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.net(x) + identity
        
    def __init__(self, layers=[32,64,128,128], n_input_channels=3, kernel_size=7):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        super().__init__()
        
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=kernel_size, padding=3, stride=1),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=1))
#            L.append(torch.nn.ReLU())
            c = l
        
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c,6)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        z = self.network(x)
        z = z.mean([2,3])
        return self.classifier(z)


class FCN(torch.nn.Module):
            
    def __init__(self, layers=[64,128], n_input_channels=3, kernel_size=7):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        
        self.relu = torch.nn.ReLU(inplace=True)
#        self.pooling = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.init_padding = 11
        self.conv_1 = torch.nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=self.init_padding, stride=1)
        self.conv_2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv_3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv_4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv_5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_6 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_7 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.convtr_1 = torch.nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_2 = torch.nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_3 = torch.nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_4 = torch.nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.bnorm_1 = torch.nn.BatchNorm2d(32)
        self.bnorm_2 = torch.nn.BatchNorm2d(32)
        self.bnorm_3 = torch.nn.BatchNorm2d(32)
        self.bnorm_4 = torch.nn.BatchNorm2d(64)
        self.bnorm_5 = torch.nn.BatchNorm2d(64)
        self.bnorm_6 = torch.nn.BatchNorm2d(128)
        self.bnorm_7 = torch.nn.BatchNorm2d(128)
        
        self.bnormtr_1 = torch.nn.BatchNorm2d(5)
        self.bnormtr_2 = torch.nn.BatchNorm2d(5)
        self.bnormtr_3 = torch.nn.BatchNorm2d(5)
        self.bnormtr_4 = torch.nn.BatchNorm2d(5)
        
        self.mp_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.res_4 = torch.nn.Conv2d(32, 64, kernel_size=1, stride=1) 
        self.res_6 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1)

        self.classifier = torch.nn.Conv2d(128, 5, kernel_size=1)
        
#        n_class = 5
#        
#        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
#        self.relu1_1 = nn.ReLU(inplace=True)
#        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
#        self.relu1_2 = nn.ReLU(inplace=True)
#        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
#
#        # conv2
#        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
#        self.relu2_1 = nn.ReLU(inplace=True)
#        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
#        self.relu2_2 = nn.ReLU(inplace=True)
#        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
#
#        # conv3
#        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
#        self.relu3_1 = nn.ReLU(inplace=True)
#        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
#        self.relu3_2 = nn.ReLU(inplace=True)
#        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
#        self.relu3_3 = nn.ReLU(inplace=True)
#        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
#
#        # conv4
#        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
#        self.relu4_1 = nn.ReLU(inplace=True)
#        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
#        self.relu4_2 = nn.ReLU(inplace=True)
#        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
#        self.relu4_3 = nn.ReLU(inplace=True)
#        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
#
#        # conv5
#        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
#        self.relu5_1 = nn.ReLU(inplace=True)
#        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
#        self.relu5_2 = nn.ReLU(inplace=True)
#        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
#        self.relu5_3 = nn.ReLU(inplace=True)
#        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
#
#        # fc6
#        self.fc6 = nn.Conv2d(512, 4096, 7)
#        self.relu6 = nn.ReLU(inplace=True)
#        self.drop6 = nn.Dropout2d()
#
#        # fc7
#        self.fc7 = nn.Conv2d(4096, 4096, 1)
#        self.relu7 = nn.ReLU(inplace=True)
#        self.drop7 = nn.Dropout2d()
#
#        self.score_fr = nn.Conv2d(4096, n_class, 1)
#        self.score_pool4 = nn.Conv2d(512, n_class, 1)
#
#        self.upscore2 = nn.ConvTranspose2d(
#            n_class, n_class, 4, stride=2, bias=False)
#        self.upscore16 = nn.ConvTranspose2d(
#            n_class, n_class, 32, stride=16, bias=False)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
#        
#        h = x
#        h = self.relu1_1(self.conv1_1(h))
#        h = self.relu1_2(self.conv1_2(h))
#        h = self.pool1(h)
#
#        h = self.relu2_1(self.conv2_1(h))
#        h = self.relu2_2(self.conv2_2(h))
#        h = self.pool2(h)
#
#        h = self.relu3_1(self.conv3_1(h))
#        h = self.relu3_2(self.conv3_2(h))
#        h = self.relu3_3(self.conv3_3(h))
#        h = self.pool3(h)
#
#        h = self.relu4_1(self.conv4_1(h))
#        h = self.relu4_2(self.conv4_2(h))
#        h = self.relu4_3(self.conv4_3(h))
#        h = self.pool4(h)
#        pool4 = h  # 1/16
#
#        h = self.relu5_1(self.conv5_1(h))
#        h = self.relu5_2(self.conv5_2(h))
#        h = self.relu5_3(self.conv5_3(h))
#        h = self.pool5(h)
#
#        h = self.relu6(self.fc6(h))
#        h = self.drop6(h)
#
#        h = self.relu7(self.fc7(h))
#        h = self.drop7(h)
#
#        h = self.score_fr(h)
#        h = self.upscore2(h)
#        upscore2 = h  # 1/16
#
#        h = self.score_pool4(pool4)
#        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
#        score_pool4c = h  # 1/16
#
#        h = upscore2 + score_pool4c
#
#        h = self.upscore16(h)
#        h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()
#        return h
                
        z_1 = self.conv_1(x)
        a_1 = self.relu(self.bnorm_1(z_1))
        z_2 = self.conv_2(a_1)
        a_2 = self.relu(self.bnorm_2(z_2)) #+ a_1
        z_3 = self.conv_3(a_2)
        a_3 = self.relu(self.bnorm_3(z_3)) #+ a_2
        z_4 = self.conv_4(a_3)
        a_4 = self.relu(self.bnorm_4(z_4)) #+ self.res_4(a_3)
        z_5 = self.conv_5(a_4)
        a_5 = self.relu(self.bnorm_5(z_5)) #+ a_4
        z_6 = self.conv_6(a_5)
        a_6 = self.relu(self.bnorm_6(z_6)) #+ self.res_6(a_5)
        z_7 = self.conv_7(a_6)
        a_7 = self.relu(self.bnorm_7(z_7)) #+ a_6

        z = self.classifier(a_7)
        
        z = self.bnormtr_1(self.convtr_1(z))
        z = self.bnormtr_2(self.convtr_2(z))
        z = self.bnormtr_3(self.convtr_3(z))
        z = self.bnormtr_4(self.convtr_4(z))
        
        #print((self.init_padding -1))
        #print(x.size(2) + (self.init_padding -1))
        z = z[:,:,int((self.init_padding -1)): int(x.size(2) + (self.init_padding -1)), int((self.init_padding -1)): int(x.size(3) + (self.init_padding -1))]
        
        return z


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
