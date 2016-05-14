
# coding: utf-8

# In[228]:

import os
import re
import glob
import hashlib
import numpy as np
from PIL import Image

def get_data():
    width = 40
    height = 40
    depth = 3
    testing_percentage = 20
    total_type = 12
    is_root = True
    
    result = {}
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    
    img_dir = 'images'
    sub_dirs = [p[0] for p in os.walk(img_dir)]

    for p in sub_dirs:
    
        file_list = []
        
        if is_root :
            is_root = False
            continue
        
        sub_dir = os.path.basename(p)
        label_name = re.sub(r'[^a-z]+', '', sub_dir.lower())
        label_index = (int)(sub_dir.split('_')[0])
    
        file_glob = os.path.join(img_dir, sub_dir, '*.jpg')
        file_list.extend(glob.glob(file_glob))
    
        for file_name in file_list:
            im = Image.open(file_name)
            im = np.array(im)
            im_n = im.reshape(width*height*depth)
        
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
            percentage_hash = (int(hash_name_hashed, 16) % (65536)) * (100 / 65536.0)
            if percentage_hash < testing_percentage:
                test_images.append(im_n)
                label = np.zeros(total_type)
                label[label_index] = 1
                test_labels.append(label)
            else:
                train_images.append(im_n)
                label = np.zeros(total_type)
                label[label_index] = 1
                train_labels.append(label)
    result = {'train_labels' : train_labels, 'train_images' : train_images, 
              'test_labels' : test_labels, 'test_images' : test_images}
    return result
