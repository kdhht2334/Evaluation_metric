__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-
# python 3.6
import os
import matplotlib.pyplot as plt
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
                #print(k+1)
#                print(data_f['{0:05}'.format(0)])
                x.append(data_f['{0:05}'.format(k)]['arousal'])
                y.append(data_f['{0:05}'.format(k)]['valence'])
                j = j + 1
#                if j >= 24:
#                    break
#            if j < 24:
#                for w in range(24-j):
#                    x.append(data_f['{0:05}'.format(k)]['arousal'])
#                    y.append(data_f['{0:05}'.format(k)]['valence'])
    return np.asarray(x), np.asarray(y)


# Load image lists
lists = []
for i in range(0,1):  # 1~121
    box = list(list_box('/home/kdh/Desktop/pytorch/AFEW-VA/aae_based/custom/' + str(i) + '/', ['.jpeg', '.png', 'JPEG']))
    paths = box_paths('/home/kdh/Desktop/pytorch/AFEW-VA/aae_based/custom/' + str(i) + '/', box)
    for i in range(len(paths)):
        path = list(list_all_files(paths[i], ['.jpeg', '.png', 'JPEG']))
        lists.append(path)
        path = []
    print('[INFO] Loaded ' + str(len(lists)) + str(i+1) + ' file lists')

X = path_to_dataset(lists)

np.save('/home/kdh/Desktop/pytorch/AFEW-VA/db/X.npz', X)

plt.imshow(X[100])  # show 100th image in X numpy array


# Load label lists (as json format)
json_lists = []
for i in range(1,13):  # 1~12
    box = list(list_box('/home/kdh/Desktop/pytorch/AFEW-VA/db/' + str(i) + '/', ['.json']))
    paths = box_paths('/home/kdh/Desktop/pytorch/AFEW-VA/db/' + str(i) + '/', box)
    for i in range(len(paths)):
        path = list(list_all_files(paths[i], ['.json']))
        json_lists.append(path)
        path = []
    print('[INFO] Loaded '  + str(len(json_lists)) + ' json file lists')
    
#json_lists_small = json_lists[:5]
y_arousal, y_valence = path_to_label(json_lists)

y_list = []
y_list.append(y_arousal)
y_list.append(y_valence)
y = np.vstack(y_list)
y = np.transpose(y, (1, 0))

np.savez('/home/kdh/Desktop/pytorch/AFEW-VA/db/y.npz', y)

#np.savez('/media/EmotDMSL/AFEW-AV/dataset/y_arousal.npz', y_arousal)
#np.savez('/media/EmotDMSL/AFEW-AV/dataset/y_valence.npz', y_valence)
    
#from pprint import pprint
#pprint(data)  # very long lines... I dont't recommend to do it :)



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
            
y1 = np.ones(1000)[:, None]
y2 = np.ones(1000)[:, None]*2
y3 = np.ones(1000)[:, None]*3
y4 = np.ones(1000)[:, None]*4
y5 = np.ones(1000)[:, None]*5
y  = np.vstack([y1, y2, y3, y4, y5])


mylist = [['subDirectory_filePath', 'label']]
for i in range(len(name_list)):
    mylist.append([name_list[i], y[i][0]])
    
import csv
with open('/home/kdh/Desktop/pytorch/AFEW-VA/aae_based/custom/all.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(mylist)
    
    
import numpy as np
np.savetxt('/home/kdh/Desktop/pytorch/AFEW-VA/db/all.csv', 
           mylist, delimiter=",", fmt='%s')



'''

    ...
    ...
    ...
    
               [247.8419526094613,
                199.65668920460985],
               [252.21717267269185,
                202.3650642798261],
               [252.47388458801538,
                208.64109283500517],
               [251.26907061392117,
                216.44151389407776],
               [247.26204799404744,
                220.222414824843],
               [242.56374988143187,
                220.72487610823984],
               [237.19236088856056,
                220.02483600506508],
               [228.9715911949036,
                215.59473324619535],
               [224.10340358856834,
                207.2558062017711],
               [238.31408676566934,
                204.62924967926497],
               [242.94451022514374,
                205.03593601380663],
               [247.14349545044251,
                204.85296887285537],
               [250.44273946288584,
                208.3077163490604],
               [247.19199020909366,
                212.28976489444065],
               [242.8471937762721,
                212.95262910398537],
               [238.04549135647358,
                212.1635336586149]],
            u'valence': 0.0}},
    u'video_id': u'001'}
    
'''



