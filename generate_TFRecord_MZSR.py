import imageio
import os
import glob
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

def augmentation(x,mode):
    if mode ==0:
        y=x

    elif mode ==1:
        y=np.flipud(x)

    elif mode == 2:
        y = np.rot90(x,1)

    elif mode == 3:
        y = np.rot90(x, 1)
        y = np.flipud(y)

    elif mode == 4:
        y = np.rot90(x, 2)

    elif mode == 5:
        y = np.rot90(x, 2)
        y = np.flipud(y)

    elif mode == 6:
        y = np.rot90(x, 3)

    elif mode == 7:
        y = np.rot90(x, 3)
        y = np.flipud(y)

    return y

def imread(path):
    img = imageio.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

def write_to_tfrecord(writer, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }))
    writer.write(example.SerializeToString())
    return

def generate_TFRecord(label_path,tfrecord_file,patch_h,patch_w,stride):
    label_list=np.sort(np.asarray(glob.glob(label_path)))

    offset=0

    fileNum=len(label_list)

    labels=[]

    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n+1), fileNum))
        label=imread(label_list[n])

        x, y, ch = label.shape
        for m in range(8):
            for i in range(0+offset,x-patch_h+1,stride):
                for j in range(0+offset,y-patch_w+1,stride):
                    patch_l = label[i:i + patch_h, j:j + patch_w]

                    if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -6.0:
                        labels.append(augmentation(patch_l,m).tobytes())

    np.random.shuffle(labels)
    print('Num of patches:', len(labels))
    print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch))

    writer = tf.io.TFRecordWriter(tfrecord_file)
    for i in range(len(labels)):
        if i % 10000 == 0:
            print('[%d/%d] Processed' % ((i+1), len(labels)))
        write_to_tfrecord(writer, labels[i])

    writer.close()

if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--labelpath', dest='labelpath', help='Path to HR images (./DIV2K_train_HR)')
    parser.add_argument('--tfrecord', dest='tfrecord', help='Save path for tfrecord file', default='train_SR_MZSR')
    options=parser.parse_args()

    labelpath=os.path.join(options.labelpath, '*.png')
    tfrecord_file = options.tfrecord + '.tfrecord'

    generate_TFRecord(labelpath, tfrecord_file,64,64,120)
    print('Done')

