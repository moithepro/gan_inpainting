import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Constants


def create_small_mask(height, width, channels=3, mask_size_ratio=Constants.SQUARE_MASK_RATIO):
    mask = np.zeros((height, width, channels))
    mask_height = int(height * mask_size_ratio)
    mask_width = int(width * mask_size_ratio)

    x1, y1 = np.random.randint(0, width - mask_width), np.random.randint(0, height - mask_height)
    x2, y2 = x1 + mask_width, y1 + mask_height
    mask[y1:y2, x1:x2, :] = 1
    return mask


def create_small_mask(height, width, x, y, channels=3, mask_size_ratio=Constants.SQUARE_MASK_RATIO):
    mask = np.zeros((height, width, channels))
    mask_height = int(height * mask_size_ratio)
    mask_width = int(width * mask_size_ratio)

    x2, y2 = x + mask_width, y + mask_height
    mask[y:y2, x:x2, :] = 1
    return mask


def create_circular_mask(height, width, channels=3, radius_ratio=Constants.CIRCLE_MASK_RATIO):
    mask = np.zeros((height, width, channels))
    radius = int(min(height, width) * radius_ratio)
    x_center, y_center = np.random.randint(radius, width - radius), np.random.randint(radius, height - radius)

    y, x = np.ogrid[:height, :width]
    mask_area = (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2
    mask[mask_area] = 1
    return mask


def create_arbitrary_mask(height, width, channels=3, ratio=Constants.ARBITRARY_MASK_RATIO):
    """
    Creates an arbitrary polygonal mask on an image.

    Parameters:
    - height: The height of the image.
    - width: The width of the image.
    - channels: Number of channels in the image. Default is 3 (RGB).
    - ratio: The proportion of the mask area to the image area. Default is 0.1 (10%).

    Returns:
    - mask: A mask with the same dimensions as the input, where the mask area is filled with ones.
    """

    # Initialize the mask with zeros
    mask = np.zeros((height, width, channels))

    # Determine the number of vertices for the polygon (between 3 and 10)
    num_vertices = np.random.randint(3, 10)

    # Generate random vertices for the polygon
    vertices = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(num_vertices)]
    vertices = np.array(vertices, dtype=np.int32)

    # Create a temporary mask to calculate the area of the polygon
    temp_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(temp_mask, [vertices], 1)

    # Calculate the area of the polygon
    polygon_area = np.sum(temp_mask)

    # Calculate the desired mask area based on the ratio
    total_area = height * width
    desired_mask_area = total_area * ratio

    # Adjust the vertices to match the desired mask area
    scale_factor = np.sqrt(desired_mask_area / polygon_area)
    vertices = vertices * scale_factor
    vertices = np.clip(vertices, 0, [width - 1, height - 1])  # Ensure vertices stay within image bounds

    # Redefine the vertices to integer values
    vertices = vertices.astype(np.int32)

    # Fill the polygon area in the mask
    cv2.fillPoly(mask, [vertices], (1, 1, 1))

    return mask


def create_random_mask(height, width, channels=3):
    mask_type = np.random.choice(['small', 'circular', 'arbitrary'])
    mask = None
    if mask_type == 'small':
        mask = create_small_mask(height, width, channels)
    elif mask_type == 'circular':
        mask = create_circular_mask(height, width, channels)
    elif mask_type == 'arbitrary':
        mask = create_arbitrary_mask(height, width, channels)
    # invert the mask before returning because I did an oopsie in the mask creation functions
    return 1 - mask


def apply_mask(image, mask):
    """
    Applies the mask to the image.
    """
    return image * mask


def calculate_metrics(original, generated):
    """
    Calculates SSIM and PSNR between original and generated images.
    """
    # Ensure both tensors are in float32 for metric calculation
    # original = tf.cast(original, tf.float32)
    # generated = tf.cast(generated, tf.float32)
    original = (original + 1.0) / 2.0  # Scale to [0, 1]
    generated = (generated + 1.0) / 2.0  # Scale to [0, 1]
    ssim = tf.image.ssim(original, generated, max_val=1.0)
    psnr = tf.image.psnr(original, generated, max_val=1.0)
    return ssim.numpy(), psnr.numpy()


def plot_images(masked, generated, original, save_path):
    """
    Plots and saves the masked, generated, and original images.
    """
    num_images = masked.shape[0]
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    for i in range(num_images):
        axes[0, i].imshow((masked[i] + 1) / 2)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Masked Image')

        axes[1, i].imshow((generated[i] + 1) / 2)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Generated Image')

        axes[2, i].imshow((original[i] + 1) / 2)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Original Image')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
