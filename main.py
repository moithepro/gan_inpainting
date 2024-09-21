import os
import tensorflow as tf


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
if os.path.exists(GENERATOR_PATH) and os.path.exists(DISCRIMINATOR_PATH):
    print("Loading pre-trained models...")
    generator = tf.keras.models.load_model(GENERATOR_PATH)
    discriminator = tf.keras.models.load_model(DISCRIMINATOR_PATH)
else:
    print("Initializing new models...")
    generator = build_generator()
    discriminator = build_discriminator()

# List of datasets in the order to be trained
datasets = ['cifar10']

for dataset_name in datasets:
    print(f"\n--- Training on {dataset_name.upper()} Dataset ---")
    dataset, total_batches = load_dataset(dataset_name)
    generator, discriminator = train(dataset, EPOCHS, dataset_name, generator, discriminator, total_batches)


# Save the trained models
print("\nSaving the trained models...")
generator.save(GENERATOR_PATH)
discriminator.save(DISCRIMINATOR_PATH)
print("Models saved successfully.")
