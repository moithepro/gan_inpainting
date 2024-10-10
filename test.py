import tensorflow as tf

from Constants import *
from utils import create_mask, apply_mask, plot_images
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os



def load_image(file_path):
    """
    Loads and preprocesses the image.
    """
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 127.5 - 1.0  # Rescale to [-1, 1]
    image = tf.expand_dims(image, axis=0)
    return image

def main():
    # Load the trained generator
    generator_path = GENERATOR_PATH
    if not os.path.exists(generator_path):
        print("Trained generator model not found. Please train the model first.")
        return
    generator = tf.keras.models.load_model(generator_path)
    print("Loaded the trained generator model.")
    
    # Open file explorer to select an image
    Tk().withdraw()  # Close the root window
    file_path = askopenfilename(title="Select an Image for Inpainting", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    
    if not file_path:
        print("No file selected.")
        return
    
    # Load and preprocess the image
    original_image = load_image(file_path)
    
    # Get user input for mask size and position (optional)
    print("Enter mask size (default is 28): ")
    user_mask_size = input()
    if user_mask_size.isdigit():
        mask_size = int(user_mask_size)
    else:
        mask_size = MASK_SIZE
    
    # Create a random or fixed mask
    print("Do you want to specify the mask position? (y/n): ")
    user_choice = input().lower()
    if user_choice == 'y':
        print(f"Enter x position (0 to {IMAGE_SIZE - mask_size}): ")
        x = int(input())
        print(f"Enter y position (0 to {IMAGE_SIZE - mask_size}): ")
        y = int(input())
        mask = create_mask(IMAGE_SIZE, mask_size, fixed_position=(x, y))
    else:
        mask = create_mask(IMAGE_SIZE, mask_size)
    
    mask = tf.expand_dims(mask, axis=0)  # Shape: (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    
    # Apply the mask
    masked_image = apply_mask(original_image, mask)
    
    # Generate the inpainted image
    generated_image = generator(masked_image, training=False)
    
    # Convert tensors to numpy arrays
    masked_image_np = masked_image.numpy()
    generated_image_np = generated_image.numpy()
    original_image_np = original_image.numpy()

    # Display the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Masked Image', 'Generated Image', 'Original Image']
    images = [masked_image_np, generated_image_np, original_image_np]

    for i in range(3):
            axes[i].imshow((images[i][0] + 1) / 2)  # Rescale to [0,1]
            axes[i].set_title(titles[i])
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
