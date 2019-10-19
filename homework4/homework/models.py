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
    def __init__(self):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        raise NotImplementedError('Detector.__init__')

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        raise NotImplementedError('Detector.forward')

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: List of detections [(class_id, score, cx, cy), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """
        raise NotImplementedError('Detector.detect')

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
