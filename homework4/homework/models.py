import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
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
#    response = list(set(response))
#    print(response)
    return response


class Detector(torch.nn.Module):
    def __init__(self, layers=[64,128], n_input_channels=3, kernel_size=7):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        
        self.relu = torch.nn.ReLU(inplace=True)
        self.init_padding = 1
        self.conv_1 = torch.nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=self.init_padding, stride=1)
        self.conv_2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv_3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv_4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv_5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_6 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_7 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_8 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_9 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.convtr_1 = torch.nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_2 = torch.nn.ConvTranspose2d(5+64, 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_3 = torch.nn.ConvTranspose2d(5+32, 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_4 = torch.nn.ConvTranspose2d(5+32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        #No Skip connections
        self.convtr_1_ns = torch.nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_2_ns = torch.nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_3_ns = torch.nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_4_ns = torch.nn.ConvTranspose2d(5, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.bnorm_1 = torch.nn.BatchNorm2d(32)
        self.bnorm_2 = torch.nn.BatchNorm2d(32)
        self.bnorm_3 = torch.nn.BatchNorm2d(32)
        self.bnorm_4 = torch.nn.BatchNorm2d(64)
        self.bnorm_5 = torch.nn.BatchNorm2d(64)
        self.bnorm_6 = torch.nn.BatchNorm2d(128)
        self.bnorm_7 = torch.nn.BatchNorm2d(128)
        self.bnorm_8 = torch.nn.BatchNorm2d(128)
        self.bnorm_9 = torch.nn.BatchNorm2d(128)
        
        self.bnormtr_1 = torch.nn.BatchNorm2d(5)
        self.bnormtr_2 = torch.nn.BatchNorm2d(5)
        self.bnormtr_3 = torch.nn.BatchNorm2d(5)
        self.bnormtr_4 = torch.nn.BatchNorm2d(3)
        
        self.mp_1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.mp_2 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.mp_3 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.res_1 = torch.nn.Conv2d(32, 32, kernel_size=1, stride=1) 
        self.res_2 = torch.nn.Conv2d(32, 32, kernel_size=1, stride=2) 
        self.res_3 = torch.nn.Conv2d(32, 32, kernel_size=1, stride=2) 
        self.res_4 = torch.nn.Conv2d(32, 64, kernel_size=1, stride=2) 
        self.res_5 = torch.nn.Conv2d(64, 64, kernel_size=1, stride=2) 
        self.res_6 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.res_7 = torch.nn.Conv2d(128, 128, kernel_size=1, stride=1) 
        self.res_8 = torch.nn.Conv2d(128, 128, kernel_size=1, stride=1) 
        self.res_9 = torch.nn.Conv2d(128, 128, kernel_size=1, stride=1) 

        self.classifier = torch.nn.Conv2d(128, 5, kernel_size=1)
        

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        z = x
        z_1 = self.conv_1(z)
        a_1 = self.relu(self.bnorm_1(z_1))
#        a_1 = F.pad(a_1, (0,int(np.ceil(z.size(2)/2 % 1)), 0, int(np.ceil(z.size(3)/2 % 1))))
            
        z = a_1
        z_2 = self.conv_2(z)
        a_2 = self.relu(self.bnorm_2(z_2)) + self.res_2(z)
#        a_2 = F.pad(a_2, (0,int(np.ceil(z.size(2)/2 % 1)), 0, int(np.ceil(z.size(3)/2 % 1))))
        
        z = a_2
        z_3 = self.conv_3(z)
        a_3 = self.relu(self.bnorm_3(z_3)) + self.res_3(z)
        a_3 = self.mp_1(a_3)
#        a_3 = F.pad(a_3, (0,int(np.ceil(z.size(2)/2 % 1)), 0, int(np.ceil(z.size(3)/2 % 1))))
        
        z = a_3
        z_4 = self.conv_4(z)
        a_4 = self.relu(self.bnorm_4(z_4)) + self.res_4(z)
#        a_4 = F.pad(a_4, (0,int(np.ceil(z.size(2)/2 % 1)), 0, int(np.ceil(z.size(3)/2 % 1))))
        
        z = a_4
        z_5 = self.conv_5(z)
        a_5 = self.relu(self.bnorm_5(z_5)) + self.res_5(z)
        a_5 = self.mp_2(a_5)
#        a_5 = F.pad(a_5, (0,int(np.ceil(z.size(2)/2 % 1)), 0, int(np.ceil(z.size(3)/2 % 1))))
        
        z = a_5
        z_6 = self.conv_6(z)
        a_6 = self.relu(self.bnorm_6(z_6)) + self.res_6(z)
#        a_6 = F.pad(a_6, (0,int(np.ceil(z.size(2)/2 % 1)), 0, int(np.ceil(z.size(3)/2 % 1))))
        
        z = a_6
        z_7 = self.conv_7(z)
        a_7 = self.relu(self.bnorm_7(z_7)) + self.res_7(z)
        a_7 = self.mp_3(a_7)
#        a_7 = F.pad(a_7, (0,int(np.ceil(z.size(2)/2 % 1)), 0, int(np.ceil(z.size(3)/2 % 1))))
        
        z = a_7
        z_8 = self.conv_8(z)
        a_8 = self.relu(self.bnorm_8(z_8)) + self.res_8(z)
#        a_7 = F.pad(a_7, (0,int(np.ceil(z.size(2)/2 % 1)), 0, int(np.ceil(z.size(3)/2 % 1))))
        
        z = a_8
        z_9 = self.conv_9(z)
        a_9 = self.relu(self.bnorm_9(z_9)) + self.res_9(z)
#        a_7 = F.pad(a_7, (0,int(np.ceil(z.size(2)/2 % 1)), 0, int(np.ceil(z.size(3)/2 % 1))))
        
        z = self.classifier(a_9)
        
        z0 = self.bnormtr_1(self.convtr_1(z))
        try:
            z = self.bnormtr_2(self.convtr_2(torch.cat((z0, a_4),1)))
            z = self.bnormtr_3(self.convtr_3(torch.cat((z, a_3),1)))
            z = self.bnormtr_4(self.convtr_4(torch.cat((z, a_2),1)))
        except:
            #no skip
            z = self.bnormtr_2(self.convtr_2_ns(z0))
            z = self.bnormtr_3(self.convtr_3_ns(z))
            z = self.bnormtr_4(self.convtr_4_ns(z))
        
        #print((self.init_padding -1))
        #print(x.size(2) + (self.init_padding -1))
        z = z[:,:,int((self.init_padding -1)): int(x.size(2) + (self.init_padding -1)), int((self.init_padding -1)): int(x.size(3) + (self.init_padding -1))]
        
        return z

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: List of detections [(class_id, score, cx, cy), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """
        flat_image = image.max(0).values
        return extract_peak(flat_image)
        
            
    def detect_with_size(self, image):
        """
           Your code here. (extra credit)
           Implement object detection here.
           @image: 3 x H x W image
           @return: List of detections [(class_id, score cx, cy, w/2, h/2), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """
        raise NotImplementedError('Detector.detect_with_size')


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    fig, axs = subplots(3, 4)
    model = load_model()
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        for c, s, cx, cy in model.detect(im):
            ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
