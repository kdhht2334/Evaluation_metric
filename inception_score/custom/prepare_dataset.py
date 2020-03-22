__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-
# python 3.6
import os
import cv2

import numpy as np
import json


def list_all_files(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or ext.lower() in extensions:
                yield joined
           
            
def list_box(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        boxes = []
        boxes.append(dirnames)
        for box in boxes:
            return box
        
        
def box_paths(directory, box):
    joined = []
    for i in range(len(box)):
        joined.append(os.path.join(directory, str(box[i])))
    return joined


def path_to_dataset(examples):
    X = []
    c = 0
    for i in range(len(examples)):
        print('[INFO] ' + str(c) + 'th file processing...')
        c += 1
        j = 0
        for path in examples[i]:
            #img = imread(path, as_grey=True)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            #img_h = cv2.equalizeHist(img)  # Histogram Equalization
            X.append(img)
            j = j + 1
            if j >= 24:
                break
        if j < 24:
            for w in range(24-j):
                X.append(img)
    return np.asarray(X)


def path_to_label(json_lists):
    x = []
    y = []
    c = 1
    for i in range(len(json_lists)):
        print('[INFO] ' + str(c) + 'th file processing...')
        c += 1
        for path in json_lists[i]:
            j = 0
            with open(path) as json_data:
                data = json.load(json_data)
                data_f = data['frames']  # to prevent unicode setting
                                         # if you don't like this...
                                         # data['frames'][u'00001'][u'arousal']
                                         # u'string' is unicode string format in UTF-8 and so on
            for k in range(len(data_f)):
                x.append(data_f['{0:05}'.format(k)]['arousal'])
                y.append(data_f['{0:05}'.format(k)]['valence'])
                j = j + 1
    return np.asarray(x), np.asarray(y)


# Load image lists
lists = []
for i in range(0,1):  # 1~121
    box = list(list_box('/your/path/custom/' + str(i) + '/', ['.jpeg', '.png', '.JPEG']))
    paths = box_paths('/your/path/custom/' + str(i) + '/', box)
    for i in range(len(paths)):
        path = list(list_all_files(paths[i], ['.jpeg', '.png', '.JPEG']))
        lists.append(path)
        path = []
    print('[INFO] Loaded ' + str(len(lists)) + str(i+1) + ' file lists')


### make csv file using list
name_list = []
for i in range(len(lists)):
    for j in range(len(lists[i])):
        if len(lists[i][j].split('/')[-1].split('.')[0]) == 1:
            name_list.append(lists[i][j][-8:])
        elif len(lists[i][j].split('/')[-1].split('.')[0]) == 2:
            name_list.append(lists[i][j][-9:])
        elif len(lists[i][j].split('/')[-1].split('.')[0]) == 3:
            name_list.append(lists[i][j][-10:])
        elif len(lists[i][j].split('/')[-1].split('.')[0]) == 4:
            name_list.append(lists[i][j][-11:])
            
y1 = np.ones(200)[:, None]
y2 = np.ones(200)[:, None]*2
y3 = np.ones(200)[:, None]*3
y4 = np.ones(200)[:, None]*4
y5 = np.ones(200)[:, None]*5
y  = np.vstack([y1, y2, y3, y4, y5])


mylist = [['subDirectory_filePath', 'label']]
for i in range(len(name_list)):
    mylist.append([name_list[i], y[i][0]])
    
import csv
with open('/your/path/custom/all.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(mylist)
    

