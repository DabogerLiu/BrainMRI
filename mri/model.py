import os
import csv
import pydicom
import numpy as np
import pickle
# import matplotlib.pyplot as plt
from skimage import transform, color
import keras
import random
import tensorflow as tf
import scipy.io

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from skimage import measure
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from keras.models import load_model
from skimage.filters import threshold_otsu
import glob
import hdf5storage
# 1. pre-defined variables

batch_size = 32           # batch size
height, width = 256 ,256  # input size
n_class = 3               # number of class 

# 2. data
path_here = '/data/shaobo/MRI/'
files = [f for f in glob.glob(path_here + "*.mat", recursive=True)]
file_folder = []
for f in files:
    file_folder.append(f)
random.shuffle(file_folder)
train_filenames = file_folder[0:2500]
valid_filenames = file_folder[2500:3000]
test_filenames = file_folder[3000:]
data_len = len(file_folder)
#print(file_folder)

def get_batch(folder, batch_size):
    while True:
        c = np.random.choice(folder, batch_size)
        count_batch = 0
        img_in_all = []
        img_seg_gt_all = []
        img_target_all = []
        for each_file in c:
            mat_file = hdf5storage.loadmat(each_file)
            img_in = mat_file['cjdata'][0][2]
            img_in =  np.array(img_in) 
            scale = (np.max(img_in)-np.min(img_in))/256
            img_in =  img_in/scale
            img_in = (np.reshape(img_in,[height, width])
            img_seg = mat_file['cjdata'][0][4]
            img_seg = np.array(img_seg)
            img_target = mat_file['cjdata'][0][0][0][0]
            img_target = np.int32(img_target)
            if img_target == 3:
                img_target = 0

            img_target = keras.utils.to_categorical(img_target, 3)
            #print(img_target)
            #plt.imshow(np.reshape(img_seg_gt,[512, 512]), cmap='gray')
            #plt.show()
            #plt.savefig('/home/CVL1/Shaobo/mri/mask.png')
            img_in = transform.resize(img_in, (height, width, 1), mode='reflect')
            img_seg = transform.resize(img_seg, (height, width, 1), mode='reflect')
            

            img_in_all.append(img_in)
            img_seg_all.append(img_seg)
            img_target_all.append(img_target)
            
            if count_batch >= batch_size-1:
                break
            count_batch += 1
            
        img_in_all = np.array(img_in_all)
        img_in_all = np.reshape(img_in_all, [batch_size, height, width,1])
        img_seg_gt_all = np.array(img_seg_all)
        img_seg_gt_all = np.reshape(img_seg_all, [batch_size, height, width,1])
        img_target_all = np.array(img_target_all)
        img_target_all = np.reshape(img_target_all, [batch_size, n_class])
        
       
        #plt.imshow(np.reshape(img_in_all[0],[height, width]), cmap='gray')
        #plt.show()
        #plt.savefig('/home/CVL1/Shaobo/mri/tumor.png')
        #plt.imshow(np.reshape(img_seg_gt_all[0],[height, width]), cmap='gray')
        #plt.show()
        #plt.savefig('/home/CVL1/Shaobo/mri/mask.png')
        #return
        yield ({'image_in': img_in_all}, \
            {'segmentation': img_seg_all, 'classification': img_target_all})

### layer / model
from keras.layers import Input, Conv2D, concatenate, add, Dense, Dropout, MaxPooling2D, Flatten, \
                          UpSampling2D, Reshape, BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D
from keras.models import Model
import keras.backend as K
from keras.utils import multi_gpu_model
#from keras.utils.vis_utils import plot_model


def unet(in_image=(height, width, 1)):
    img_in = Input(shape = in_image, name='image_in')
    
    conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_1')(img_in)
    conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_2')(conv0)
    conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_3')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2),name='down0')(conv0)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_1')(pool0)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_2')(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_3')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_1' )(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_2')(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_3')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='down2')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_1')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_2')(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='down3')(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_1')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_2')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_3')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),name='down4')(drop4)

    down_4_f = Flatten(name='down_2_flat')(pool4)

    down_classsify = Dense(512,activation='relu',name='classify_1')(down_4_f)
    down_classsify = Dropout(0.5)(down_classsify)
    down_classsify = Dense(128,activation='relu',name='classify_2')(down_classsify)
    down_classsify = Dropout(0.5)(down_classsify)
    classification = Dense(3,activation='sigmoid',name='classification')(down_classsify)



    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_1')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_2')(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_3')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_1')(drop5)
    up6 = UpSampling2D(size = (2,2),name = 'up_1')(up6)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_2')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name ='conv6_3')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_1')(conv6)
    up7 = UpSampling2D(size = (2,2),name = 'up2')(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_2')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_3')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_1')(conv7)
    up8 = UpSampling2D(size = (2,2),name ='up3')(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_2')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_3')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_1')(conv8)
    up9 = UpSampling2D(size = (2,2),name = 'up4')(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_2')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_3')(conv9)
    
    up10 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_1')(conv9)
    up10 = UpSampling2D(size = (2,2),name = 'up5')(up10)
    merge10 = concatenate([conv0,up10], axis = 3)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_2')(merge10)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_3')(conv10)
    
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_4')(conv10)

    segmentation = Conv2D(1, 1, activation = 'sigmoid', name='segmentation')(conv10)

    model = Model(inputs = img_in, outputs = [segmentation, classification])

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    return model
    

import keras.backend as K
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

def seg_loss(y_true,y_pred):
        return 0.8 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.2* iou_loss(y_true, y_pred)


def compile_model():
    opti = keras.optimizers.Adadelta(lr=0.05)
    model = unet()
    model.compile(optimizer=opti,
                  loss=[seg_loss, 'binary_crossentropy'],
                  loss_weights=[0.75, 0.25],
                  metrics = {'classification':'accuracy'})

    return model

def train(epoch=5):
    model = compile_model()

    history = model.fit_generator(get_batch(train_filenames,batch_size), validation_data = get_batch(valid_filenames,batch_size), \
                                 steps_per_epoch=int(data_len / batch_size), epochs=epoch, validation_steps= int(500 / batch_size))
    model.save('/home/CVL1/Shaobo/mri/MRI_seg_Shaobo.h5')
    with open('/home/CVL1/Shaobo/mri/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

from matplotlib import pyplot as plt
def plot_loss():
    with open('/home/CVL1/Shaobo/mri/history.pkl', 'rb') as file_pi:
        history = pickle.load(file_pi)

    plt.figure(figsize=(12, 4))
    plt.subplot(141)
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Valid loss")
    plt.legend()
    plt.subplot(142)
    plt.plot(history["segmentation_loss"], label="segmentation_loss")
    plt.plot(history["val_segmentation_loss"], label="val_segmentation_loss")
    #plt.plot(history["segmentation_iou_score"],label = "segmentation_iou_score")
    #plt.plot(history["val_segmentation_iou_score"],label = "val_segmentation_iou_score")
    plt.legend()
    plt.subplot(143)
    plt.plot(history["classification_loss"], label="classification_loss")
    plt.plot(history["val_classification_loss"], label="val_classification_loss")
    plt.legend()
    plt.subplot(144)
    plt.plot(history["classification_acc"],label = "classification_acc")
    plt.plot(history["val_classification_acc"],label = "val_classification_acc")
    plt.legend()
    plt.show()
    plt.savefig('/home/CVL1/Shaobo/mri/training.png')
    return

from keras.models import load_model
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations
import numpy as np
import matplotlib.cm as cm


def plot_img():
    
    model = load_model('/home/CVL1/Shaobo/mri/MRI_seg_Shaobo.h5')
    gb = get_batch(test_filenames,n_test_samples)
    abatch = next(gb)
    imgs = abatch[0]['image_in']
    msks = abatch[1]['segmentation']
    labels = abatch[1]['classification']

    f, axarr = plt.subplots(25, 20, figsize=(64,64))
    axarr = axarr.ravel()
    axidx = 0
    IOU_score = []
    precision_score = []
    axarr = axarr.ravel()
    count_Seg = []
    count_SegIoU = []
    count_class = []
    
    for i in range(n_test_samples):

        img = imgs[i]
        msk = msks[i]
        labels = labels[i]

        img1 = np.reshape(img,[height, width])
        axarr[axidx].imshow(img1)

        msk = np.reshape(msk,[height, width])
        comp = msk > 0.5 / 1.
        comp1 = measure.label(comp)
        predictionString = ''
        for region in measure.regionprops(comp1):
            # retrieve x, y, height and width
            y11, x11, y12, x12 = region.bbox
            height1 = y12 - y11
            width1 = x12 - x11
            axarr[axidx].add_patch(patches.Rectangle((x11,y11),width1,height1,linewidth=2,edgecolor='b',facecolor='none'))

    ### segment
        predImg = np.reshape(imgs[i],[1, height, width, 1])
        seg_pre, class_pre = model.predict(predImg)

        pred = np.reshape(seg_pre,[height, width])
        pred = (pred > 0.5) / 1.
        pred1 =  measure.label(pred)

        for region in measure.regionprops(pred1):
            # retrieve x, y, height and width
            y21, x21, y22, x22 = region.bbox
            height2 = y22 - y21
            width2 = x22 - x21
            axarr[axidx].add_patch(patches.Rectangle((x21,y21),width2,height2,linewidth=2,edgecolor='r',facecolor='none'))
        axidx += 1
        intersection = np.logical_and(comp1, pred1)
        union = np.logical_or(comp1, pred1)
        iou_score =np.sum(intersection) / np.sum(union)
        precision =np.sum(intersection) / np.sum(comp1)

        if (np.isnan(precision) == True):
            precision=0
            count_Seg.append(0)
        else:
            count_Seg.append(1)
        precision_score.append(precision)

        if (np.isnan(iou_score) == True):
            iou_score=0
            count_SegIoU.append(0)
        else:
            count_SegIoU.append(1)
        IOU_score.append(iou_score) 
    plt.show()
    plt.savefig('/home/CVL1/Shaobo/PneSeg/model/plot_valid.png')
    print(IOU_score)
    print("mean_IoU",sum(IOU_score)/sum(count_SegIoU))

    print("mean_Precision",sum(precision_score)/sum(count_Seg))   
    print("Test_Accuracy",sum(count_class)/n_test_samples)
    #print("count_label",count_label)
    #print("count_pred",count_pred)
    return

def plot_saliency_map():
    model = load_model('/home/CVL1/Shaobo/PneSeg/model/RSNA_seg_Shaobo.h5', custom_objects={'seg_loss': seg_loss})
    gb = get_batch(test_filenames, n_test_samples)
    abatch = next(gb)
    imgs = abatch[0]['image_in']
    msks = abatch[1]['segmentation']
    labels = abatch[1]['classification']

if __name__ == "__main__":
    #print("===============")
    gb = get_batch(test_filenames, 1)
    #train(epoch=20)
    #if send_mail(mailto_list, "main.py on GPUstation finished", "Auto-mail from my python codes: segmentation result is ready. -Daboger"):  print("====notification sent.====")
   # plot_loss()
   # plot_img()
