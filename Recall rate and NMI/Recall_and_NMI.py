#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np, os, sys, pandas as pd, csv, random, datetime

import torch, torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl

from sklearn import metrics
from sklearn import cluster

import faiss

import losses as losses


def f1_score(model_generated_cluster_labels, target_labels, feature_coll, computed_centroids):
    """
    NOTE: MOSTLY ADAPTED FROM https://github.com/wzzheng/HDML on Hardness-Aware Deep Metric Learning.
    Args:
        model_generated_cluster_labels: np.ndarray [n_samples x 1], Cluster labels computed on top of data embeddings.
        target_labels:                  np.ndarray [n_samples x 1], ground truth labels for each data sample.
        feature_coll:                   np.ndarray [n_samples x embed_dim], total data embedding made by network.
        computed_centroids:             np.ndarray [num_cluster=num_classes x embed_dim], cluster coordinates
    Returns:
        float, F1-score
    """
    from scipy.special import comb

    d = np.zeros(len(feature_coll))
    for i in range(len(feature_coll)):
        d[i] = np.linalg.norm(feature_coll[i,:] - computed_centroids[model_generated_cluster_labels[i],:])

    labels_pred = np.zeros(len(feature_coll))
    for i in np.unique(model_generated_cluster_labels):
        index = np.where(model_generated_cluster_labels == i)[0]
        ind = np.argmin(d[index])
        cid = index[ind]
        labels_pred[index] = cid


    N = len(target_labels)

    #Cluster n_labels
    avail_labels = np.unique(target_labels)
    n_labels     = len(avail_labels)

    #Count the number of objects in each cluster
    count_cluster = np.zeros(n_labels)
    for i in range(n_labels):
        count_cluster[i] = len(np.where(target_labels == avail_labels[i])[0])

    #Build a mapping from item_id to item index
    keys     = np.unique(labels_pred)
    num_item = len(keys)
    values   = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])


    #Count the number of objects of each item
    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[labels_pred[i]]
        count_item[index] = count_item[index] + 1

    #Compute True Positive (TP) plus False Positive (FP) count
    tp_fp = 0
    for k in range(n_labels):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    #Compute True Positive (TP) count
    tp = 0
    for k in range(n_labels):
        member = np.where(target_labels == avail_labels[k])[0]
        member_ids = labels_pred[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    #Compute  False Positive (FP) count
    fp = tp_fp - tp

    #Compute False Negative (FN) count
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)
    fn = count - tp

    # compute F measure
    beta = 1
    P  = tp / (tp + fp)
    R  = tp / (tp + fn)
    F1 = (beta*beta + 1) * P * R / (beta*beta * P + R)

    return F1


def eval_metrics_one_dataset(model, test_dataloader, device, k_vals, opt):
    """
    Compute evaluation metrics on test-dataset, e.g. NMI, F1 and Recall @ k.
    Args:
        model:              PyTorch network, network to compute evaluation metrics for.
        test_dataloader:    PyTorch Dataloader, dataloader for test dataset, should have no shuffling and correct processing.
        device:             torch.device, Device to run inference on.
        k_vals:             list of int, Recall values to compute
        opt:                argparse.Namespace, contains all training-specific parameters.
    Returns:
        F1 score (float), NMI score (float), recall_at_k (list of float), data embedding (np.ndarray)
    """
    torch.cuda.empty_cache()

    _ = model.eval()
    n_classes = len(test_dataloader.dataset.avail_classes)

    with torch.no_grad():
        ### For all test images, extract features
        target_labels, feature_coll = [],[]
        final_iter = tqdm(test_dataloader, desc='Computing Evaluation Metrics...')
        image_paths= [x[0] for x in test_dataloader.dataset.image_list]
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            feature_coll.extend(out.cpu().detach().numpy().tolist())

        target_labels = np.hstack(target_labels).reshape(-1,1)
        feature_coll  = np.vstack(feature_coll).astype('float32')

        torch.cuda.empty_cache()

        ### Set Faiss CPU Cluster index
        cpu_cluster_index = faiss.IndexFlatL2(feature_coll.shape[-1])
        kmeans            = faiss.Clustering(feature_coll.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000

        ### Train Kmeans
        kmeans.train(feature_coll, cpu_cluster_index)
        computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, feature_coll.shape[-1])

        ### Assign feature points to clusters
        faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        faiss_search_index.add(computed_centroids)
        _, model_generated_cluster_labels = faiss_search_index.search(feature_coll, 1)

        ### Compute NMI
        NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))


        ### Recover max(k_vals) nearest neighbours to use for recall computation
        faiss_search_index  = faiss.IndexFlatL2(feature_coll.shape[-1])
        faiss_search_index.add(feature_coll)
        _, k_closest_points = faiss_search_index.search(feature_coll, int(np.max(k_vals)+1))
        k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

        ### Compute Recall
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(target_labels)
            recall_all_k.append(recall_at_k)

        ### Compute F1 Score
        F1 = f1_score(model_generated_cluster_labels, target_labels, feature_coll, computed_centroids)

    return F1, NMI, recall_all_k, feature_coll
