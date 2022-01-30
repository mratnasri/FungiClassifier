import os
from glob import glob
import cv2
from numpy import mean, std
import tensorflow.keras as keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

#gram_img_dir = '../../Gram Stain Images'
gram_img_dir = '../../Gram Stain Images_ver3'
M = 150
N = 150


#train_total = sum(train_n)


def img_crops(path, folder, label):
    img_names = glob(path)
    print("number of images in ", label, ": ", len(img_names))
    images = []
    img_num = 0
    crop_all_dir = folder+'_all/'+label
    os.makedirs(crop_all_dir, exist_ok=True)
    crop_dir = folder+'/'+label
    os.makedirs(crop_dir, exist_ok=True)
    crop_contrast_stretched_dir = folder+'_contrast_stretched/'+label
    os.makedirs(crop_contrast_stretched_dir, exist_ok=True)

    for fn in img_names:
        img_num = img_num+1
        img = cv2.imread(fn)
        #img = load_img(fn)
        img = np.array(img)
        if (img.shape[0] > img.shape[1]):  # height, width
            img = np.rot90(img, 1)  # Counter-Clockwise
        # cv2.imshow('img',img)
        r = 1200.0 / img.shape[1]
        dim = (1200, int(img.shape[0] * r))  # width, height
        # perform the actual resizing of the image
        img = cv2.resize(img, dim)
        images.append(img)

        # Convert into grayscale
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray_img = np.array(gray_img, dtype='uint8')

        crops = [img[x:x+M, y:y+N]
                 for x in range(0, img.shape[0]-M+1, M) for y in range(0, img.shape[1]-N+1, N)]
        print("no. of crops in ", img_num, ": ", len(crops))
        i = 0
        num = 0
        for crop in crops:
            num = num+1
            # Convert into grayscale
            gray_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            print("Non zero pixels: ", cv2.countNonZero(gray_img))
            crop_name = label+'-' + \
                str('{:02}'.format(img_num))+'_' + \
                str('{:03}'.format(num))+'.jpg'
            cv2.imwrite(crop_all_dir+'/'+crop_name, crop)
            mask = np.ones(crop.shape[0:2], dtype="uint8")
            mask[gray_img <= 20] = 0
            if (cv2.countNonZero(mask) >= 80/100*N*M):
                print("true")
                i = i+1
                # temp=crop
                cv2.imwrite(folder+'/'+label+'/'+label+'-' +
                            str('{:02}'.format(img_num))+'_'+str('{:03}'.format(i))+'.jpg', crop)
                #crop = crop.astype(np.uint8)
                # print(crop.dtype)
                crop = ((crop-np.min(crop))/(np.max(crop)-np.min(crop)))*255
                crop = crop.astype(np.uint8)
                #crop = crop.astype('float32')
                # cv2.imshow('crop',crop)
                # cv2.waitKey(0)
                cv2.imwrite(folder+'_contrast_stretched/'+label+'/'+label+'-'+str('{:02}'.format(
                    img_num))+'_'+str('{:03}'.format(i))+'_contrast_stretched.jpg', crop)

    return images


for label in os.listdir(gram_img_dir):
    # train_categories.append(label)
    path = gram_img_dir + "/" + label + "/*.jpg"
    #train_n.append(len(os.listdir(gram_img_dir + "/" + label)))
    folder = '../../Gram Stain Images_ver3_Crops'
    img_crops(path, folder, label)

print('done')
