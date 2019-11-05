import torch
from .utils import spatial_argmax
from torch import nn
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=0.4, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    
    H,W = heatmap.size()
    max_map = F.max_pool2d(heatmap[None,None], kernel_size=max_pool_ks, stride=1, padding=max_pool_ks//2)
    mask = (heatmap >= max_map) & (heatmap > min_score)
    
    mask.squeeze_(0).squeeze_(0).size()
    local_maxima = heatmap[mask]
    
    top_k = torch.topk(local_maxima, min(len(local_maxima),max_det), sorted=True)
    indices = (mask == True).nonzero()
    
    response = []
    for i in range(len(top_k.values)):
        response.append((top_k.values[i].item(), indices[top_k.indices[i]][1].item(), (indices[top_k.indices[i]][0].item())))

    return response

class Planner(torch.nn.Module):
    
    class conv_block(torch.nn.Module):
        def __init__(self, channel_in, channel_out, stride=2, kernel_size=3, dilation=1):
            super().__init__()
            self.c1 = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=kernel_size//2)
            self.b1 = nn.BatchNorm2d(channel_out)
            self.c2 = nn.Conv2d(channel_out, channel_out, kernel_size=kernel_size, dilation=dilation, stride=1, padding=kernel_size//2)
            self.b2 = nn.BatchNorm2d(channel_out)
            
            self.downsample = None
            if channel_in != channel_out or stride != 1:
                self.downsample = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride, dilation=dilation)
            
        def forward(self, x):
            self.activation = F.relu(self.b2(self.c2(self.b1(self.c1(x))))) #consider adding relus between
            identity = x
            if self.downsample != None:
                identity = self.downsample(identity)
            return self.activation + identity
            
    class upconv_block(torch.nn.Module):
        
        def __init__(self, channel_in, channel_out, stride=2, kernel_size=3, dilation=1):    
            super().__init__()
            self.upsample = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, dilation=dilation, output_padding=1)
        
        def forward(self, x, output_pad=False):
            return F.relu(self.upsample(x))
            
    def __init__(self, layers=[32,32,64,64,128], image_size=(96,128)):
        super().__init__()

        """
        Your code here
        """
        self.image_size = image_size
        c = 3        
        self.network = torch.nn.ModuleList()
        for l in layers:
            kernel_size = 7 if c == 3 else 3
            stride = 1 if c == 3 else 2
            self.network.append(self.conv_block(c, l, stride, kernel_size, 1))
            c = l
        
        self.upnetwork = torch.nn.ModuleList()
        self.upnetwork.append(self.upconv_block(c, layers[-2]))
        c = layers[-2]
        for l in reversed(layers[:-2]):
            self.upnetwork.append(self.upconv_block(c * 2, l, 2, 3, 1)) # x2 input because of skip
            c = l
        self.classifier = nn.Conv2d(c, 1, kernel_size=1)
        

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        
        ##Add preprocessing
        x = img
        activations = []
        for i, layer in enumerate(self.network):
            z = layer(x)
            activations.append(z)
            x = z
        z = self.upnetwork[0](x)
        for i, layer in enumerate(self.upnetwork[1:]):
            x = torch.cat([z[:,:, :activations[-2-i].size(2), :activations[-2-i].size(3)], activations[-2-i]], dim=1)
            z = layer(x)
        heatmap = F.sigmoid(self.classifier(z))
        heatmap = heatmap.squeeze(1)
        
        output = spatial_argmax(heatmap)
        # print(output)
        #Resize to image size
        if output.is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        img_size = torch.tensor(list(reversed(self.image_size)), dtype=torch.float, device=device)
        # print(output)
        output = ((output / 2.0) + 0.5) * img_size
        # print(output, img_size)
        return output
        


def save_model(model, label=None):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        if label is None:
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
        else:
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner{}.th'.format(label)))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model(label=None):
    from torch import load
    from os import path
    r = Planner()
    if label is None:
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    else:
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner{}.th'.format(label)), map_location='cpu'))
    return r

# def load_model():
#     from torch import load
#     from os import path
#     r = Planner()
#     r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
#     return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
