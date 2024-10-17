import tensorflow as tf

import incremental_saver
import utils
from Constants import *
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
from models import build_generator


def load_image(file_path):
    """
    Loads and preprocesses the image.
    """
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 127.5 - 1.0  # Rescale to [-1, 1]
    image = tf.expand_dims(image, axis=0)
    return image


def extract_feature_maps(generator, masked_image):
    """
    Extracts feature maps from specific layers of the encoder part of the generator.
    """
    encoder_layers = ['leaky_re_lu_1', 'leaky_re_lu_2',
                      'leaky_re_lu_3']  # Names of the layers from which to extract features
    intermediate_outputs = [generator.get_layer(name).output for name in encoder_layers]
    intermediate_model = tf.keras.Model(inputs=generator.input, outputs=intermediate_outputs)
    feature_maps = intermediate_model(masked_image, training=False)
    return feature_maps


def plot_feature_maps(feature_maps):
    """
    Plots feature maps from specific layers.
    """
    for i, feature_map in enumerate(feature_maps):
        num_filters = feature_map.shape[-1]
        num_cols = 8
        num_rows = num_filters // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        fig.suptitle(f'Feature Maps from Encoder Layer {i + 1}')

        for r in range(num_rows):
            for c in range(num_cols):
                ax = axes[r, c]
                ax.imshow(feature_map[0, :, :, r * num_cols + c], cmap='viridis')
                ax.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    # Load the trained generator
    generator_path = incremental_saver.get_last_models()[0]
    if not os.path.exists(generator_path):
        print("Trained generator model not found. Please train the model first.")
        return
    generator = tf.keras.models.load_model(generator_path)
    print("Loaded the trained generator model.")

    # Open file explorer to select an image
    Tk().withdraw()  # Close the root window
    file_path = "free-images.jpg" # askopenfilename(title="Select an Image for Inpainting", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])

    if not file_path:
        print("No file selected.")
        return

    # Load and preprocess the image
    original_image = load_image(file_path)

    # Get user input for mask size and position (optional)
    print("Enter mask size ratio: ")
    user_mask_size = input()
    if user_mask_size.isdigit():
        mask_size = float(user_mask_size)
    else:
        mask_size = SQUARE_MASK_RATIO

    # Create a random or fixed mask
    print("Do you want to specify the mask position? (y/n): ")
    user_choice = input().lower()
    if user_choice == 'y':
        print(f"Enter x position (0 to {IMAGE_SIZE - mask_size}): ")
        x = int(input())
        print(f"Enter y position (0 to {IMAGE_SIZE - mask_size}): ")
        y = int(input())
        mask = utils.create_small_mask_with_position(IMAGE_SIZE, IMAGE_SIZE, x, y, mask_size_ratio=mask_size)
    else:
        mask = utils.create_small_mask(IMAGE_SIZE, IMAGE_SIZE, mask_size_ratio=mask_size)

    # convert mask to float32
    mask = tf.cast(mask, tf.float32)

    soft_mask = utils.create_soft_mask(mask)  # Create a soft mask for blending before expanding dims
    mask = tf.expand_dims(mask, axis=0)  # Shape: (1, IMAGE_SIZE, IMAGE_SIZE, 3)

    # Apply the mask
    masked_image = utils.apply_mask(original_image, mask)

    # Generate the inpainted image
    generated_image = generator(masked_image, training=False)

    # Blend the generated image with the original image

    blended_image = utils.blend_images(original_image, generated_image, soft_mask)

    # Extract feature maps from encoder layers
    feature_maps = extract_feature_maps(generator, masked_image)

    # Convert tensors to numpy arrays
    masked_image_np = masked_image.numpy()
    generated_image_np = generated_image.numpy()
    original_image_np = original_image.numpy()
    blended_image_np = blended_image.numpy()

    # Display the images
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    titles = ['Masked Image', 'Generated Image', 'Original Image', 'Generated Image after blending process']
    images = [masked_image_np, generated_image_np, original_image_np, blended_image_np]

    for i in range(4):
        axes[i].imshow((images[i][0] + 1) / 2)  # Rescale to [0,1]
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Plot feature maps from encoder layers
    plot_feature_maps(feature_maps)


if __name__ == "__main__":
    main()