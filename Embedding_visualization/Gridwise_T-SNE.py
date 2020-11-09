import random
import numpy as np
import json
import matplotlib.pyplot
import pickle
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.manifold import TSNE

import torch, argparse
import random, glob, os
import numpy as np
from tqdm import tqdm

import cv2
from scipy.spatial.distance import pdist, squareform

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *

# $ git clone https://github.com/Quasimondo/RasterFairy
# $ pip install .
import rasterfairy


# --------------
# Load functions
# --------------

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


# -------------------------
# Initialize configurations
# -------------------------

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



# --------------------------------
# Load pre-trained model & weights
# --------------------------------

# Backbone Model
trained_dataset = 'inshop'
model_name      = 'bn_inception'
length = 6400  # CUB: 2500 | Cars: 4900 | SOP: 8100 | In-shop: 6400

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
if trained_dataset == 'cub':
    print("CUB!")
    path = 'path/to/cub_weights.pth'
elif trained_dataset == 'cars':
    print("Cars!")
    path = 'path/to/cars_weights.pth'
elif trained_dataset == 'sop':
    print("SOP!")
    path = 'path/to/sop_weights.pth'
elif trained_dataset == 'inshop':
    print("In-shop Clothes!")
    path = 'path/to/inshop_weights.pth'

ww = torch.load(path, map_location="cuda:0")['model_state_dict']
# for key in list(ww.keys()):
#     if 'module.' in key:
#         ww[key.replace('module.', '')] = ww[key]
#         del ww[key]
model.load_state_dict(ww)
model.eval()


# ----------------
# Extract features
# ----------------

if trained_dataset == 'cub' or trained_dataset == 'cars':
    # folder_list = glob.glob(r'W:\DML\datasets\cub200\images_qualitative\*')
    folder_list = glob.glob(r'W:\DML\datasets\cars196\images\*')[98:]
    # for i in tqdm(range(50)):
    for i in tqdm(range(len(folder_list))):
        if i == 0:
            img_0_list = sorted(glob.glob(folder_list[i]+'/*'), key=os.path.getmtime)
        else:
            img_0_list.extend(sorted(glob.glob(folder_list[i]+'/*'), key=os.path.getmtime))
elif trained_dataset == 'sop':
    root_path = r'W:\DML\datasets\online_products\Stanford_Online_Products'
    f = open(r'W:\DML\datasets\online_products\Stanford_Online_Products\Ebay_test.txt')
    new_list = []
    for _ in f:
        new_list.append(root_path + '\\' + f.readline().split(' ')[-1][:-1])
    
    for i in tqdm(range(len(new_list))):
        if i == 0:
            img_0_list = glob.glob(new_list[i])
        else:
            img_0_list.extend(glob.glob(new_list[i]))
elif trained_dataset == 'inshop':
    print("\n")
    print("dd")
    root_path = r'W:\DML\datasets\in-shop'
    f = open(r'W:\DML\datasets\in-shop\Eval\list_eval_partition.txt')
    new_list = []
    for _ in f:
        if f.readline().split(' ')[0].split('/')[0] == 'img':
            if f.readline().split(' ')[-7] != 'train':
                new_list.append(root_path + '\\' + f.readline().split(' ')[0])
            
    for i in tqdm(range(len(new_list))):
        if i == 0:
            img_0_list = glob.glob(new_list[i])
        else:
            img_0_list.extend(glob.glob(new_list[i]))
        
images = img_0_list[:length]


img_0_pr_list = []
for i in tqdm(range(len(images))):
    img_0_pr_list.append(img_postprocessing(images[i]))
    
vec_0_list = []
for i in tqdm(range(len(img_0_pr_list))):
    vec_0_list.append(model(img_0_pr_list[i]).detach().cpu().numpy())

images = images[:length]
vec_0_list = vec_0_list[:length]

# ---------------------
# Visualize using T-SNE
# ---------------------

X = np.array(vec_0_list)[:,0,:]
tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

# tx, ty = tsne[:,0], tsne[:,1]
# tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
# ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

# width = 4000
# height = 3000
# max_dim = 100

# full_image = Image.new('RGBA', (width, height))
# for img, x, y in zip(images, tx, ty):
#     tile = Image.open(img)
#     rs = max(1, tile.width/max_dim, tile.height/max_dim)
#     tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
#     full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

# matplotlib.pyplot.figure(figsize = (16,12))
# imshow(full_image)


# -----------------------
# Grid-wise visualization
# -----------------------

# Note that nx * ny = len(images)
nx = int(np.sqrt(len(images)))
ny = int(np.sqrt(len(images)))

# assign to grid
grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))


tile_width = 100
tile_height = 100

full_width = tile_width * nx
full_height = tile_height * ny
aspect_ratio = float(tile_width) / tile_height

grid_image = Image.new('RGB', (full_width, full_height))

for img, grid_pos in zip(images, grid_assignment[0]):
    idx_x, idx_y = grid_pos
    x, y = tile_width * idx_x, tile_height * idx_y
    tile = Image.open(img)
    tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
    if (tile_ar > aspect_ratio):
        margin = 0.5 * (tile.width - aspect_ratio * tile.height)
        tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
    else:
        margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
        tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
    tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
    grid_image.paste(tile, (int(x), int(y)))

matplotlib.pyplot.figure(figsize = (16,16), dpi=500)
plt.axis('off')
imshow(grid_image)
