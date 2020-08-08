import tensorflow as tf
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

CIFAR_MEAN = [125.3, 123.0, 113.9]
CIFAR_STD = [63.0, 62.1, 66.7]

HEIGHT = 224
WIDTH = 224

class Preprocess:
    """
    Preprocesses the images before submitting to the network.

    Args:
      data_format: channels_first or channels_last
      train: whether or not to do data augmentation for training
    """

    def __init__(self, data_format, train):
        self._data_format = data_format
        self._train = train

    def __call__(self, image):
        image = tf.cast(image, tf.float32)
        if self._train:
            image = tf.image.random_flip_left_right(image)
            image = self.random_jitter(image)

        image = (image - CIFAR_MEAN) / CIFAR_STD

        if self._data_format == 'channels_first':
            image = tf.transpose(image, [2, 0, 1])

        return image

    def random_jitter(self, image):
        # resize to have a bit of random cropping
        image = tf.image.resize_with_crop_or_pad(
            image, HEIGHT + 32, WIDTH + 32)

        image = tf.image.random_crop(image, size=[HEIGHT, WIDTH, 3])

        return image

# Following https://www.tensorflow.org/tutorials/load_data/images as a tutorial

def get_label(file_path):
    # TODO: implement this given the data we do have
    pass

def decode_img(img):
    return tf.image.decode_jpeg(img, channels=3)

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def create_dataset(buffer_size, batch_size, data_format, data_dir="GarageImages", do_augment=True):
    """
    Creates a tf.data Dataset from a data directory

    Args:
      buffer_size: shuffle buffer size
      batch_size: Batch size
      data_format: channels_first or channels_last
      data_dir: directory to load the dataset from
      do_augment: whether or not to augment the training data
    Returns:
      train_dataset
    """

    preprocess_train = Preprocess(data_format, do_augment)
    list_ds = tf.data.Dataset.list_files(data_dir + '/*', shuffle=False)
    train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=buffer_size)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds
