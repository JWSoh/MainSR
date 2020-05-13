import tensorflow as tf
from config import *

class dataLoader(config):
    def __init__(self):
        super(dataLoader, self).__init__(args)
        self.BUFFER_SIZE=1000
        self.augmentation=True

    def augment(self, labels, images,
                horizontal_flip=True,
                rotate=True):
        with tf.name_scope('augmentation'):
            shp = tf.shape(images)
            batch_size, height, width = shp[0], shp[1], shp[2]
            width = tf.cast(width, tf.float32)
            height = tf.cast(height, tf.float32)

            transforms = []
            identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
            if horizontal_flip:
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
                flip_transform = tf.convert_to_tensor(
                    [-1., 0., width-1, 0., 1., 0., 0., 0.], dtype=tf.float32)
                transforms.append(
                    tf.where(coin,
                             tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if rotate:
                angles = tf.to_float(tf.random_uniform([batch_size], 0, 4, dtype=tf.int32))
                angles = angles*np.pi/2
                transforms.append(
                    tf.contrib.image.angles_to_projective_transforms(
                        angles, height, width))

            if transforms:
                images = tf.contrib.image.transform(
                    images,
                    tf.contrib.image.compose_transforms(*transforms),
                    interpolation='BILINEAR')

                labels = tf.contrib.image.transform(
                    labels,
                    tf.contrib.image.compose_transforms(*transforms),
                    interpolation='BILINEAR')

        return labels, images

    def _parse_function(self, example_proto):
        keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                            'label': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        img = parsed_features['image']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img = tf.reshape(img, [self.HEIGHT // self.scale, self.WIDTH // self.scale, self.CHANNEL])

        label = parsed_features['label']
        label = tf.divide(tf.cast(tf.decode_raw(label, tf.uint8), tf.float32), 255.)
        label = tf.reshape(label, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        return label, img

    def load_tfrecord(self):
        dataset = tf.data.TFRecordDataset(self.TF_RECORD_PATH)
        dataset = dataset.map(self._parse_function)

        dataset = dataset.shuffle(self.BUFFER_SIZE)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train, image_train = iterator.get_next()

        if self.augmentation:
            label_train, image_train = self.augment(label_train, image_train)

        return label_train, image_train