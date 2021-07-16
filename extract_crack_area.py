
#%% 
import numpy as np # 일반적인 행렬 연산 Library 
import matplotlib.pyplot as plt # 그림을 그려주는 Library 
import cv2 # 이미지처리 Library
import os # 경로 관련 Library 
from glob import glob 
import slidingwindow as sw
from tqdm import tqdm
from datetime import datetime

 # %%

data_dir = '/home/soojin/UOS-SSaS Dropbox/05. Data/02. Training&Test/concrete_damage_autocon/as_cityscape'
data_save_dir = '/home/soojin/UOS-SSaS Dropbox/05. Data/02. Training&Test/02. Autolabel-train/crack/cityscape_210716'

img_dir = os.path.join(data_dir, 'leftImg8bit', 'train')
label_dir = os.path.join(data_dir, 'gtFine', 'train')

img_save_dir = os.path.join(data_save_dir, 'leftImg8bit', 'train')
label_save_dir = os.path.join(data_save_dir, 'gtFine', 'train')


# %%

img_list = glob(os.path.join(img_dir, '*.png'))
# %%

for img_path in tqdm(img_list) : 
    img_basename = os.path.basename(img_path)
    img_filename = img_basename[:-16]
    label_path = os.path.join(label_dir, img_filename + '_gtFine_labelIds.png')

    img = cv2.imread(img_path)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    # Code Reference : https://github.com/adamrehn/slidingwindow
    # Generate the set of windows, with a 128-pixel max window size and 10% overlap
    windows = sw.generate(label, sw.DimOrder.HeightWidthChannel, 128, 0.1)
    # Do stuff with the generated windows
    for window in windows:
        # print(window.indices())
        label_subset = label[window.indices()]
        num_pxls_in_object = np.sum(label_subset == 1)

        if num_pxls_in_object > 300: 
            img_subset = img[window.indices()]
            # 저장 경로 설정
            current_time = datetime.now().time()

            save_file_name = img_filename + str(current_time)

            save_img_name = save_file_name + '_leftImg8bit.png'
            save_label_name = save_file_name + '_gtFine_labelIds.png'

            save_img_path = os.path.join(img_save_dir, save_img_name)
            save_label_path = os.path.join(label_save_dir, save_label_name)
            
            cv2.imwrite(save_img_path, img_subset)
            cv2.imwrite(save_label_path, label_subset)


        




# %%

