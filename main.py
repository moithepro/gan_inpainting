import os
import tensorflow as tf

import incremental_saver

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set for GPUs")
    except RuntimeError as e:
        print("Error setting memory growth:", e)

from train import train
from data_loader import load_dataset
from models import build_generator, build_discriminator
from Constants import *




# Ensure results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# Paths for saving/loading models


# Initialize or load models

generator_path, discriminator_path = incremental_saver.get_last_models()

if generator_path is not None and discriminator_path is not None:
    print("Loading pre-trained models...")
    print(f"Generator: {generator_path}")
    print(f"Discriminator: {discriminator_path}")
    generator = tf.keras.models.load_model(generator_path)
    discriminator = tf.keras.models.load_model(discriminator_path)
else:
    print("Initializing new models...")
    generator = build_generator()
    discriminator = build_discriminator()

# List of datasets in the order to be trained
datasets = ['coco']

for dataset_name in datasets:
    print(f"\n--- Training on {dataset_name.upper()} Dataset ---")
    dataset, total_batches = load_dataset(dataset_name)
    generator, discriminator = train(dataset, EPOCHS, dataset_name, generator, discriminator, total_batches)

# 44
# Save the trained models
print("\nSaving the trained models...")
incremental_saver.save_models(generator, discriminator)
print("Models saved successfully.")
