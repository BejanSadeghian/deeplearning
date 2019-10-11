import torch


class CNNClassifier(torch.nn.Module):
    
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(n_output),
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
        
    def __init__(self, layers=[32,64,128], n_input_channels=3, kernel_size=7):
        """
        best so far is 32,64,64,128
        Your code here
        """
        super().__init__()
        
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=kernel_size, padding=3, stride=1),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=1))
            L.append(torch.nn.ReLU())
            c = l
        
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c,6)
        
    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        z = self.network(x)
        z = z.mean([2,3])
        return self.classifier(z)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
