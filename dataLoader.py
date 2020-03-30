import tensorflow as tf
from config import *

class dataLoader(config):
    def __init__(self):
        super(dataLoader, self).__init__(args)
        self.BUFFER_SIZE=1000
        self.augmentation=True

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
        
        if self.augmentation:
            mode = tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)

            if tf.random.uniform(()) > 0.5:
                img=tf.image.flip_left_right(img)
                label=tf.image.flip_left_right(label)

            img = tf.image.rot90(img, mode)
            label = tf.image.rot90(label, mode)
            
        return label, img

    def load_tfrecord(self):
        dataset = tf.data.TFRecordDataset(self.TF_RECORD_PATH)
        dataset = dataset.map(self._parse_function)

        dataset = dataset.shuffle(self.BUFFER_SIZE)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train, image_train = iterator.get_next()

        return label_train, image_train