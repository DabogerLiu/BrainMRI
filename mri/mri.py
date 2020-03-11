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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
from attention_module import attach_attention_module
from keras.utils import CustomObjectScope

# 1. pre-defined variables
batch_size = 64           # batch size
height, width = 128,128  # input size
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
data_len = len(train_filenames)

def get_batch(folder,batch_size):
    while True:
        c = np.random.choice(folder, batch_size*4)
        
        count_batch = 0
        
        img_in_all = []
        img_seg_all = []
        img_target_all = []
        for each_file in c:
            mat_file = hdf5storage.loadmat(each_file)
            img_in = mat_file['cjdata'][0][2]
            img_in =  np.array(img_in)
            #scale = (np.max(img_in)-np.min(img_in))/256
            #img_in = np.int32(img_in/scale)

            img_seg = mat_file['cjdata'][0][4]
            img_seg = np.array(img_seg)
            img_target = mat_file['cjdata'][0][0][0][0]
            img_target = np.int32(img_target)

            if img_target == 3:
                img_target = 0
            

            img_in = transform.resize(img_in, (height, width, 1), mode='reflect')
            img_seg = transform.resize(img_seg, (height, width, 1), mode='reflect')
            img_target = keras.utils.to_categorical(img_target, 3)

            img_in_all.append(img_in)
            img_seg_all.append(img_seg)
            img_target_all.append(img_target)
            
            if count_batch >= batch_size-1:
                break
            count_batch += 1
            
        img_in_all = np.array(img_in_all)
        img_in_all = np.reshape(img_in_all, [batch_size, height, width, 1])
        img_seg_all = np.array(img_seg_all)
        img_seg_all = np.reshape(img_seg_all, [batch_size, height, width, 1])
        img_target_all = np.array(img_target_all)
        img_target_all = np.reshape(img_target_all, [batch_size, n_class])

        yield ({'image_in': img_in_all}, \
              {'segmentation': img_seg_all, 'classification': img_target_all})

### layer / model
from keras.layers import Input, Conv2D, concatenate, add, Dense, Dropout, MaxPooling2D, Flatten, \
                          UpSampling2D, Reshape, BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, AveragePooling2D
from keras.models import Model
import keras.backend as K
from keras.utils import multi_gpu_model
#from keras.utils.vis_utils import plot_model

def unet(in_image=(height, width, 1)):
    img_in = Input(shape = in_image, name='image_in')
    
    #conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_1')(img_in)
    #conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_2')(conv0)
    #conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_3')(conv0)
    #conv0 = Dropout(0.3)(conv0)
    #pool0 = MaxPooling2D(pool_size=(2, 2),name='down0')(conv0)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_1')(img_in)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_2')(conv1)
    #conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_3')(conv1)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_1' )(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_2')(conv2)
    #conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_3')(conv2)
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='down2')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_1')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_2')(conv3)
    #conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_3')(conv3)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='down3')(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_1')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_2')(conv4)
    #conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_3')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),name='down4')(drop4)

    down_4_f = Flatten(name='down_2_flat')(pool4)

    down_classsify = Dense(512,activation='relu',name='classify_1')(down_4_f)
    down_classsify = Dropout(0.4)(down_classsify)
    down_classsify = Dense(128,activation='relu',name='classify_2')(down_classsify)
    down_classsify = Dropout(0.4)(down_classsify)
    classification = Dense(3,activation='sigmoid',name='classification')(down_classsify)



    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_1')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_2')(conv5)
    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_3')(conv5)
    drop5 = Dropout(0.2)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_1')(drop5)
    up6 = UpSampling2D(size = (2,2),name = 'up_1')(up6)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_2')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name ='conv6_3')(conv6)
    drop6 = Dropout(0.2)(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_1')(drop6)
    up7 = UpSampling2D(size = (2,2),name = 'up2')(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_2')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_3')(conv7)
    drop7 = Dropout(0.3)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_1')(drop7)
    up8 = UpSampling2D(size = (2,2),name ='up3')(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_2')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_3')(conv8)
    drop8 = Dropout(0.2)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_1')(drop8)
    up9 = UpSampling2D(size = (2,2),name = 'up4')(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_2')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_3')(conv9)
    conv9 = Dropout(0.3)(conv9)
    
    #up10 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_1')(drop9)
    #up10 = UpSampling2D(size = (2,2),name = 'up5')(up10)
    #merge10 = concatenate([conv0,up10], axis = 3)
    #conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_2')(merge10)
    #conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_3')(conv10)
    #drop10 = Dropout(0.6)(conv10)
    
    #conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_4')(conv10)

    segmentation = Conv2D(1, 1, activation = 'sigmoid', name='segmentation')(conv9)

    model = Model(inputs = img_in, outputs = [segmentation, classification])
    model.summary()

    return model
    
from keras.utils import plot_model

def ResCBAM(in_image=(height, width, 1)):

    img_in = Input(shape = in_image, name='image_in')
    img_in_b = img_in
    
    x0 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_0')(img_in_b)
    y0 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_1')(img_in_b)
    y0 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_2')(y0)
    y0 = attach_attention_module(y0, attention_module='cbam_block')
    conv0= add([x0,y0])
    pool1 = MaxPooling2D(pool_size=(2, 2),name='down0')(conv0)
    
    x1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_0')(pool1)
    #x1 = BatchNormalization(x1)
    y1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_1')(pool1)
    #y1 = BatchNormalization(y1)
    y1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_2')(y1)
    #y1 =  BatchNormalization(y1)
    y1 = attach_attention_module(y1, attention_module='cbam_block')
    
    conv1= add([x1,y1])
    pool2 = MaxPooling2D(pool_size=(2, 2),name='down1')(conv1)
    
    x2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_0')(pool2)
   # x2 = BatchNormalization(x2)
    y2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_1')(pool2)
    #y2 = BatchNormalization(y2)
    y2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_2')(y2)
    #y2 =  BatchNormalization(y2)
    y2 = attach_attention_module(y2, attention_module='cbam_block')
    conv2= add([x2,y2])
    pool3 = MaxPooling2D(pool_size=(2, 2),name='down2')(conv2)
    
    x3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_0')(pool3)
    #x3 = BatchNormalization(x3)
    
    y3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_1')(pool3)
    #y3 = BatchNormalization(y2)
    y3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_2')(y3)
   # y3 =  BatchNormalization(y2)
    y3 = attach_attention_module(y3, attention_module='cbam_block')
    
    conv3= add([x3,y3])
    pool3 = AveragePooling2D(pool_size=2,name='down3' )(conv3)
    down_4_f = Flatten(name='down_2_flat')(pool3)
    
    classification = Dense(3,activation='sigmoid',name='classification')( down_4_f )



    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_1')(pool3)
    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_2')(conv5)
    drop5 = Dropout(0.2)(conv5)
    
    x6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_0')(drop5)
        
    y6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_1')(drop5)
    y6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_2')(y6)
    y6 = attach_attention_module(y6, attention_module='cbam_block')
    xy6 = add([x6,y6])
    up6 = UpSampling2D(size = (2,2),name = 'up6')(xy6)
    merge6 = concatenate([conv3,up6], axis = 3)
    
    
    x7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv7_0')(merge6)
     
    y7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv7_1')(merge6)
    y7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv7_2')(y7)
    y7 = attach_attention_module(y7, attention_module='cbam_block')
    up7= add([x7,y7])
    up7 = UpSampling2D(size = (2,2),name = 'up7')(up7)
    merge7 = concatenate([conv2,up7], axis = 3)
    
    x8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv8_0')(up7)
    
    y8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv8_1')(up7)
    y8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv8_2')(y8)
    y8 = attach_attention_module(y8, attention_module='cbam_block')
    up8= add([x8,y8])
    up8 = UpSampling2D(size = (2,2),name = 'up8')(up8)
    merge8 = concatenate([conv1,up8], axis = 3)
    
    
    
    x9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv9_0')(merge8)
    
    y9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv9_1')(merge8)
    y9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv9_2')(y9)
    y9 = attach_attention_module(y9, attention_module='cbam_block')
    XY9= add([x9,y9])
    up9 = UpSampling2D(size = (2,2),name ='Up9')(XY9)
    merge9 = concatenate([conv0,up9], axis = 3)
    
    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_4')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_5')(conv9)
    segmentation = Conv2D(1, 1, activation = 'sigmoid', name='segmentation')(merge9)

    model = Model(inputs = img_in, outputs = [segmentation, classification])
    model.summary()
    plot_model(model, to_file='model.png')
    return model

import keras.backend as K

def compile_model():

    #model = unet()
    model = ResCBAM()
    opti1 = keras.optimizers.Adam(lr=0.0005)
    
    model.compile(optimizer=opti1,
                  loss=['binary_crossentropy', 'categorical_crossentropy'],
                  loss_weights=[0.5, 0.5],
                  metrics = {'classification':'accuracy'})

    return model

def train(epoch=5):
    model = compile_model()

    history = model.fit_generator(get_batch(train_filenames,batch_size), validation_data = get_batch(valid_filenames,batch_size), \
                                 steps_per_epoch=int(data_len / batch_size), epochs=epoch, validation_steps= int(len(valid_filenames) / batch_size))
    model.save('/home/CVL1/Shaobo/mri/h5File/mri.h5')
    with open('/home/CVL1/Shaobo/mri/history/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

from matplotlib import pyplot as plt
def plot_loss():
    with open('/home/CVL1/Shaobo/mri/history/history.pkl', 'rb') as file_pi:
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
from skimage.filters import threshold_otsu
def plot_img():
    
    model = load_model('/home/CVL1/Shaobo/mri/h5File/mri.h5')
    gb = get_batch(test_filenames,100)
    abatch = next(gb)
    imgs = abatch[0]['image_in']
    msks = abatch[1]['segmentation']
    labels = abatch[1]['classification']

    plt.figure(figsize=(60, 20))
    n = 40
    for i in range(n):

        img = imgs[i]
        msk = msks[i]
        print(labels[i])

        label = np.where(labels[i] == 1.)[0]
        print(label)
        

        img1 = np.reshape(img,[height, width])
        msk = np.reshape(msk,[height, width])

        predImg = np.reshape(imgs[i],[1, height, width, 1])
        seg_pre, class_pre = model.predict(predImg)
        pred = np.reshape(seg_pre,[height, width])
        pred = (pred > 0.5) / 1.
        pred =  measure.label(pred)

        class_pre = np.where(labels[i] == np.max(labels[i]))

        diff = np.abs(msk-pred)


        ax = plt.subplot(4, n, i+1)
        plt.imshow(img1,cmap='gray')
        ax.set_title(label.astype(str))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i+1 + n)
        plt.imshow(msk, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title("Mask")

        
        ax = plt.subplot(4, n, i +1 + 2*n)
        plt.imshow(pred,cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(class_pre[0].astype(str))

        ax = plt.subplot(4, n, i +1 + 3*n)
        plt.imshow(diff,cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title("Difference")
    plt.show()
    plt.savefig('/home/CVL1/Shaobo/mri/outputs/plot_valid.png')

    
    return
    
import vis
import vis.visualization
import vis.utils
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
from keras import activations

def plot_saliency():
    with CustomObjectScope({'attach_attention_module':attach_attention_module}):
        model = load_model('/home/CVL1/Shaobo/mri/h5File/mri.h5')
    model.summary()
    gb = get_batch(valid_filenames,5)
    abatch = next(gb)
    imgs = abatch[0]['image_in']
    msks = abatch[1]['segmentation']
    labels = abatch[1]['classification']
    plt.figure(figsize=(40,20))
    n = 3
    for i in range(n):
        img = imgs[i]
    
        msk = msks[i]
        class_label = labels[i]
        img1 = np.reshape(img,[height, width])
        msk = np.reshape(msk,[height, width])
        comp = msk > 0.65 / 1.
        comp1 = measure.label(comp)
    
        predImg = np.reshape(imgs[i],[1, height, width, 1])
        seg_pre, class_pre = model.predict(predImg)
        pred = np.reshape(seg_pre,[height, width])
        pred = (pred > 0.65) / 1.
        pred1 =  measure.label(pred)
        class_idx = np.where(class_label == 1.)[0]
    
        ax1 = plt.subplot(4, n, i+1)
        ax1.imshow(img1)
        ax1.set_title('Seg_Result '+ str(class_idx) )
    #ax1.get_xaxis().set_visible(False)
    #ax1.get_yaxis().set_visible(False)
   
        print(class_label)
        print(class_idx[0])
        layer_idx = utils.find_layer_idx(model, 'classification')
        print(layer_idx)
        model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(model, custom_objects = {'attach_attention_module': attach_attention_module })
        img2 = resize(imgs[i],(height, width))
    
        grads = visualize_saliency(model, layer_idx, filter_indices= class_idx[0], seed_input = img2)
    #grads = visualize_cam(model1, layer_idx, filter_indices= class_idx[0], seed_input = img2, backprop_modifier='guided')
        ax2 = plt.subplot(4, n, i+1+n)
        ax2.imshow(grads,cmap='jet')
        ax2.set_title('Saliency Map')
    #ax4.get_xaxis().set_visible(False)
   # ax4.get_yaxis().set_visible(False)
    
      
       
        model.layers[layer_idx].activation = keras.activations.linear
        model = utils.apply_modifications(model, custom_objects = {'attach_attention_module':attach_attention_module})
    #print(layer_idx)
    #model.layers[layer_idx].activation = activations.linear
        penultimate_layer = utils.find_layer_idx(model, 'down3')
        print(penultimate_layer)
    #model = utils.apply_modifications(model)
        img2 = resize(imgs[i],(height, width))
       
        grads = visualize_cam(model, layer_idx, filter_indices= class_idx[0], seed_input = img2, penultimate_layer_idx = penultimate_layer)
        ax3 = plt.subplot(4, n, i+1+2*n)
        ax3.imshow(grads, cmap='jet', alpha=0.8)
        
        ax4 = plt.subplot(4, n, i+1 + 3*n)
        plt.imshow(msk, cmap='gray')
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        ax4.set_title("Mask")
    
           
               
    plt.show()
    plt.savefig('/home/CVL1/Shaobo/mri/outputs/test1.jpg')

if __name__ == "__main__":
    print("===============")
    train(epoch=80)
    plot_loss()
    plot_img()
    plot_saliency()
