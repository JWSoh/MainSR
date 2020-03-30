import tensorflow as tf
import matplotlib.pyplot as plt
import os
from config import *
import dataLoader

conf1 = config(args)
conf_tf = tf.ConfigProto()
conf_tf.gpu_options.per_process_gpu_memory_fraction = 0.8

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = conf1.GPU


plt.rcParams['image.interpolation']='nearest'

print(tf.__version__)

dataLoader=dataLoader.dataLoader()
X, Y = dataLoader.load_tfrecord()

'''For Tensorflow Version 1.x'''
with tf.Session(config=conf_tf) as sess:
    img, label = sess.run([X, Y])
    print('img shape:', img.shape)
    print('label shape:', label.shape)

    plt.figure()
    for j in range(16):
        plt.subplot(4,4,j+1)
        plt.imshow(img[j])
        plt.axis('off')


    plt.figure()
    for j in range(16):
        plt.subplot(4,4,j+1)
        plt.imshow(label[j])
        plt.axis('off')
    plt.show()