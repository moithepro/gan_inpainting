import tensorflow as tf
from tensorflow.keras import optimizers #type: ignore
import os
import numpy as np

import incremental_saver
from Constants import *

from utils import calculate_metrics, plot_images
from data_loader import data_generator
from models import build_generator, build_discriminator
import time
from tqdm import tqdm
from tensorflow.keras.applications import VGG16
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))


def perceptual_loss(real_images, generated_images):
    # Extract features from the real and generated images using VGG16
    vgg_outputs_real = vgg(real_images)
    vgg_outputs_fake = vgg(generated_images)

    # Calculate perceptual loss over masked areas (L2 loss here)
    return tf.reduce_mean(tf.square(vgg_outputs_real - vgg_outputs_fake))


# Enable mixed precision (optional)
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

def train(dataset, epochs, dataset_name, generator=None, discriminator=None, total_batches=None):
    """
    Trains the GAN on the provided dataset.
    """
    if generator is None:
        generator = build_generator()
    if discriminator is None:
        discriminator = build_discriminator()
    
    # Learning Rate Schedulers
    lr_schedule_G = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True
    )
    lr_schedule_D = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True
    )
    
    # Optimizers with learning rate scheduling
    optimizer_G = optimizers.Adam(learning_rate=lr_schedule_G, beta_1=0.5)
    optimizer_D = optimizers.Adam(learning_rate=lr_schedule_D, beta_1=0.5)
    
    # Wrap optimizers for mixed precision (if enabled)
    # optimizer_G = mixed_precision.LossScaleOptimizer(optimizer_G, loss_scale='dynamic')
    # optimizer_D = mixed_precision.LossScaleOptimizer(optimizer_D, loss_scale='dynamic')
    
    # Loss Functions
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    mae_loss = tf.keras.losses.MeanAbsoluteError()


    # Training Step Function
    @tf.function
    def train_step(masked_images, real_images, masks):
        with tf.device('/GPU:0'):
            # Ensure tensors are on the GPU
            masked_images = tf.identity(masked_images)
            real_images = tf.identity(real_images)
            masks = tf.identity(masks)
            
            # Labels
            batch_size = tf.shape(masked_images)[0]

            with tf.GradientTape(persistent=True) as tape:
                # Generator Forward Pass
                generated_images = generator(masked_images, training=True)
                
                # Discriminator Forward Pass
                real_output = discriminator(real_images, training=True)
                fake_output = discriminator(generated_images, training=True)

                # Convert to float32 for loss calculation because of mixed precision we need to ensure the output is
                # in the same dtype as all the other tensors
                # real_output = tf.cast(real_output, tf.float32)
                # fake_output = tf.cast(fake_output, tf.float32)
                # generated_images = tf.cast(generated_images, tf.float32)
                # masks = tf.cast(masks, tf.float32)

                # Calculate Losses
                real_labels = tf.ones_like(real_output)  # Broadcast directly to match the shape of real_output
                fake_labels = tf.zeros_like(fake_output)  # Same for fake labels
                d_loss_real = bce_loss(real_labels, real_output)
                d_loss_fake = bce_loss(fake_labels, fake_output)
                d_loss = d_loss_real + d_loss_fake
                
                g_loss_GAN = bce_loss(real_labels, fake_output)
                g_loss_L1 = mae_loss(real_images * (1 - masks), generated_images * (1 - masks))
                ssim_loss = tf.reduce_mean(1 - tf.image.ssim(real_images * masks, generated_images * masks, max_val=1.0))
                g_loss_perceptual = perceptual_loss(real_images, generated_images)

                g_loss = g_loss_GAN + 100 * g_loss_L1 + 10 * ssim_loss + 0.05 * g_loss_perceptual

            # Calculate Gradients
            gradients_of_generator = tape.gradient(g_loss, generator.trainable_variables)
            gradients_of_discriminator = tape.gradient(d_loss, discriminator.trainable_variables)
            
            # Apply Gradients
            optimizer_G.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            optimizer_D.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        return d_loss, g_loss, generated_images
    
    # Training Loop
    with tf.device('/GPU:0'):
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            print(f'Starting epoch {epoch}/{epochs} for {dataset_name}')
            
            progress_bar = tqdm(data_generator(dataset), total=total_batches, desc=f'Epoch {epoch}/{epochs}', unit='batch')
            
            for masked_images, real_images, masks in progress_bar:
                d_loss, g_loss, generated_images = train_step(masked_images, real_images, masks)
                progress_bar.set_postfix({
                    'D_loss': f'{d_loss.numpy():.4f}',
                    'G_loss': f'{g_loss.numpy():.4f}'
                })

            # Save Checkpoints
            if epoch % SAVE_INTERVAL == 0:
                incremental_saver.save_models(generator, discriminator)

            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f'Epoch {epoch} completed in {epoch_duration:.2f} seconds.')
            print(f'Discriminator Loss: {d_loss.numpy():.4f}, Generator Loss: {g_loss.numpy():.4f}')
            
            # Save Generated Images and Calculate Metrics
            test_masked_images = masked_images[:5]
            test_real_images = real_images[:5]
            test_masks = masks[:5]
            test_generated_images = generator(test_masked_images, training=False)
            
            # Calculate Metrics
            ssim, psnr = calculate_metrics(test_real_images, test_generated_images)
            print(f'SSIM: {np.mean(ssim):.4f}, PSNR: {np.mean(psnr):.4f}')
            
            # Plot and Save Images
            save_path = os.path.join(RESULTS_DIR, f'{dataset_name}_epoch_{epoch}.png')
            plot_images(test_masked_images.numpy(), test_generated_images.numpy(), test_real_images.numpy(), save_path)
               
    return generator, discriminator
