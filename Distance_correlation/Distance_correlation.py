import torch, argparse
import random
import numpy as np

import cv2
from scipy.spatial.distance import pdist, squareform

# from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *


def img_postprocessing(path):
    img = cv2.imread(path)
    img = np.expand_dims(cv2.resize(img, (224, 224)), axis=0).transpose(0,3,1,2)
    
    from torchvision import transforms, utils
    img_norm = transforms.Normalize((104, 117, 128),(1, 1, 1))(torch.from_numpy(img[0]))
    #img_norm = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))(torch.from_numpy(img[0]))
    img_norm = img_norm.unsqueeze_(0).type(torch.cuda.FloatTensor)
    return img_norm


def distance_correlation(A, B):
    #https://en.wikipedia.org/wiki/Distance_correlation
    #Input
    # A: the first variable
    # B: the second variable
    # The numbers of samples in the two variables must be same.
    #Output
    # dcor: the distance correlation of the two samples

    n = A.shape[0]
    if B.shape[0] != A.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(A))
    b = squareform(pdist(B))
    T1 = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    T2 = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    #Use Equation 2 to calculate distance covariances.
    dcov_T1_T2 = (T1 * T2).sum() / float(n * n)
    dcov_T1_T1 = (T1 * T1).sum() / float(n * n)
    dcov_T2_T2 = (T2 * T2).sum() / float(n * n)

    #Equation 1 in the paper.
    dcor = np.sqrt(dcov_T1_T2) / np.sqrt(np.sqrt(dcov_T1_T1) * np.sqrt(dcov_T2_T2))
    return dcor


seed = 1278  # 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'  
    + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
)
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR', 
    default='../logs',
    help = 'Path to log folder'
)
parser.add_argument('--dataset', 
    default='cub',
    help = 'Training dataset, e.g. cub, cars, SOP, Inshop'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default = 180, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--epochs', default = 80, type = int,
    dest = 'nb_epochs',
    help = 'Number of training epochs.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 4, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--model', default = 'bn_inception',
    help = 'Model for training'
)
parser.add_argument('--loss', default = 'Proxy_Anchor',
    help = 'Criterion for training'
)
parser.add_argument('--optimizer', default = 'adamax',
    help = 'Optimizer setting'
)
parser.add_argument('--lr', default = 1e-4, type =float,
    help = 'Learning rate setting'
)
parser.add_argument('--weight-decay', default = 1e-4, type =float,
    help = 'Weight decay setting'
)
parser.add_argument('--lr-decay-step', default = 10, type =int,
    help = 'Learning decay step setting'
)
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float,
    help = 'Learning decay gamma setting'
)
parser.add_argument('--alpha', default = 32, type = float,
    help = 'Scaling Parameter setting'
)
parser.add_argument('--mrg', default = 0.1, type = float,
    help = 'Margin parameter setting'
)
parser.add_argument('--IPC', type = int,
    help = 'Balanced sampling, images per class'
)
parser.add_argument('--warm', default = 1, type = int,
    help = 'Warmup training epochs'
)
parser.add_argument('--bn-freeze', default = 1, type = int,
    help = 'Batch normalization parameter freeze'
)
parser.add_argument('--l2-norm', default = 1, type = int,
    help = 'L2 normlization'
)
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)


# os.chdir('../../research-ms-loss/resource/datasets/')
# data_root = os.getcwd()

# Backbone Model
model_name = 'bn_inception'
if model_name == 'googlenet':
    model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
elif model_name == 'bn_inception':
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
elif model_name == 'resnet50':
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)


# Load pre-trained weights

path = '/path/to/.pth file'
ww = torch.load(path, map_location="cuda:0")['model_state_dict']
# for key in list(ww.keys()):
#     if 'module.' in key:
#         ww[key.replace('module.', '')] = ww[key]
#         del ww[key]
model.load_state_dict(ww)
model.eval()


img1 = img_postprocessing('/path/to/imgae file')
img2 = img_postprocessing('/path/to/imgae file')

    
vec1 = model(img1).detach().cpu().numpy()
vec2 = model(img2).detach().cpu().numpy()

diff1 = np.abs((vec1 - vec2).mean())  # intra-class
dcov1 = distance_correlation(vec1.transpose(1,0), vec2.transpose(1,0))  # intra-class
print("L1 distance is {}".format(diff1))
print("Distance correlation (DC) is {}".format(dcov1))

