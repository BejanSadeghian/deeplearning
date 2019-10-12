import torch
import torch.nn.functional as F


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
    
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(n_output),
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
        
        self.conv_1 = torch.nn.Conv2d(n_input_channels, 32, kernel_size=kernel_size, padding=3, stride=1)
        self.conv_2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_6 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_7 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.bnorm_1 = torch.nn.BatchNorm2d(32)
        self.bnorm_2 = torch.nn.BatchNorm2d(32)
        self.bnorm_3 = torch.nn.BatchNorm2d(32)
        self.bnorm_4 = torch.nn.BatchNorm2d(64)
        self.bnorm_5 = torch.nn.BatchNorm2d(64)
        self.bnorm_6 = torch.nn.BatchNorm2d(128)
        self.bnorm_7 = torch.nn.BatchNorm2d(128)


        self.res_4 = torch.nn.Conv2d(32, 64, kernel_size=1, stride=1) 
        self.res_6 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1)

        self.classifier = torch.nn.Conv2d(128, 5, kernel_size=1)

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

                
        z_1 = self.conv_1(x)
        a_1 = self.relu(self.bnorm_1(z_1))
        z_2 = self.conv_2(a_1)
        a_2 = self.relu(self.bnorm_2(z_2)) + a_1
        z_3 = self.conv_3(a_2)
        a_3 = self.relu(self.bnorm_3(z_3)) + a_2
        z_4 = self.conv_4(a_3)
        a_4 = self.relu(self.bnorm_4(z_4)) + self.res_4(a_3)
        z_5 = self.conv_5(a_4)
        a_5 = self.relu(self.bnorm_5(z_5)) + a_4
        z_6 = self.conv_6(a_5)
        a_6 = self.relu(self.bnorm_6(z_6)) + self.res_6(a_5)
        z_7 = self.conv_7(a_6)
        a_7 = self.relu(self.bnorm_7(z_7)) + a_6


        z = self.classifier(a_7)
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
