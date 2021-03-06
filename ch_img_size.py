
# coding: utf-8

# In[ ]:

import os
from PIL import Image
import glob
img_dir = '/project/project_iii/images'
sub_dirs = [p[0] for p in os.walk(img_dir)]
is_root = True
new_size = (32, 32)
for p in sub_dirs:
    if is_root :
        is_root = False
        continue
    sub_dir = os.path.basename(p)
    file_glob = os.path.join(img_dir, sub_dir, '*.jpg')
    img_files = glob.glob(file_glob)
    for img_file in img_files:
        img_name = os.path.basename(img_file)
        img_n_name = img_name.split('.')[0]
        im = Image.open(img_file)
        nim = im.resize(new_size, Image.BILINEAR)
        new_path = p+'/'+img_n_name+'.jpg'
        nim.save(new_path)

