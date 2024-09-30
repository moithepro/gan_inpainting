import tensorflow as tf
import tensorflow_datasets as tfds
from utils import create_mask
from Constants import *

BUFFER_SIZE = 1000

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_rotation(image, 0.1)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image

def load_dataset(name):
    """
    Loads and preprocesses the specified dataset.
    """
    if name == 'mnist':
        dataset = tfds.load('mnist', split='train', as_supervised=True)
        dataset = dataset.map(lambda x, y: (tf.image.resize(tf.image.grayscale_to_rgb(x), [IMAGE_SIZE, IMAGE_SIZE]), y))
    elif name == 'emnist':
        dataset = tfds.load('emnist/letters', split='train', as_supervised=True)
        dataset = dataset.map(lambda x, y: (tf.image.resize(tf.image.grayscale_to_rgb(x), [IMAGE_SIZE, IMAGE_SIZE]), y))
    elif name == 'gtsrb':
        dataset = tfds.load('gtsrb', split='train', as_supervised=True)
        dataset = dataset.map(lambda x, y: (tf.image.resize(x, [IMAGE_SIZE, IMAGE_SIZE]), y))
    elif name == 'cifar10':
        dataset = tfds.load('cifar10', split='train', as_supervised=True)
        dataset = dataset.map(lambda x, y: (tf.image.resize(x, [IMAGE_SIZE, IMAGE_SIZE]), y))
    elif name == 'imagenet':
        dataset = tfds.load('imagenet2012', split='train', as_supervised=True)
        dataset = dataset.map(lambda x, y: (tf.image.resize(x, [IMAGE_SIZE, IMAGE_SIZE]), y))
    elif name == 'coco':
        dataset = tfds.load('coco/2017', split='train', as_supervised=True)
        dataset = dataset.map(lambda x, y: (tf.image.resize(x, [IMAGE_SIZE, IMAGE_SIZE]), y))
    elif name == 'celeba':
        dataset = tfds.load('celeb_a', split='train', as_supervised=True)
        dataset = dataset.map(lambda x, y: (tf.image.resize(x, [IMAGE_SIZE, IMAGE_SIZE]), y))
    else:
        raise ValueError('Dataset not recognized.')
    
    # Normalize images to [-1, 1]
    dataset = dataset.map(lambda x, y: ((x / 127.5) - 1.0, y))
    
    # Use image as both input and target
    dataset = dataset.map(lambda x, y: (x, x))
    
    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Get total number of batches
    total_batches = tf.data.experimental.cardinality(dataset).numpy()
    
    return dataset, total_batches


def data_generator(dataset):
    """
    Generator that yields masked and augmented images along with masks.
    """
    for batch in dataset:
        images = batch[0]
        # Create masks
        masks = tf.map_fn(lambda img: create_mask(IMAGE_SIZE, MASK_SIZE), images)
        # Apply masks
        masked_images = images * masks
        yield (masked_images, images, masks)
