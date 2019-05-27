import cv2
import os
import pdb
import numpy as np
import nibabel as nib
from sklearn.preprocessing import LabelBinarizer


IMG_ROOT = 'D:/MICCAI_BraTS17_Data_Training/seg'
IMG_PATH = 'D:/MICCAI_BraTS17_Data_Training/HGG/Brats17_2013_2_1/Brats17_2013_2_1_flair.nii.gz'
IMG_OUTPUT_ROOT = 'D:/done3'

def nii2jpg_label(img_path, output_root):
    img_name = (img_path.split('/')[-1]).split('.')[0]
    output_path = os.path.join(output_root, img_name)
    try:
        os.mkdir(output_root)
    except:
        pass
    try:
        os.mkdir(output_path)
    except:
        pass
    img = nib.load(img_path)
    img = (img.get_fdata())[:,:,:]
    img = img*50
    img = img.astype(np.uint16)

    for i in range(img.shape[2]):
        filename = os.path.join(output_path, img_name+'_'+str(i)+'.jpg')
        gray_img = img[:,:,i]
        # color_img = np.expand_dims(gray_img, 3)
        # color_img = np.concatenate([color_img, color_img, color_img], 2)

        # COLOR LABELING
        #if (color_img.max() == 0):
        color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        color_img = (color_img / 255).astype(np.uint8)
        color_img = cv2.fastNlMeansDenoisingColored(color_img, None, 10, 10, 7, 21)
        cv2.imwrite(filename, color_img)
        #else:
            # c255 = np.expand_dims(np.ones(gray_img.shape)*255, 3)
            # c0 = np.expand_dims(np.zeros(gray_img.shape), 3)
            # color = np.concatenate([c0,c0,c255], 2)
            # color_img = color_img.astype(np.float32) + color
            # color_img = (color_img / color_img.max()) *255
            # color_img = cv2.fastNlMeansDenoisingColored(color_img, None, 10, 10, 7, 21)
            # cv2.imwrite(filename, color_img)

for path in os.listdir(IMG_ROOT):
    print(path)
    if path[0] == '.':
        continue
    nii2jpg_label((IMG_ROOT+'/'+path), IMG_OUTPUT_ROOT)
