import tensorflow as tf
import matplotlib.pyplot as plt

def create_mask(image_size, mask_size, fixed_position=None):
    """
    Creates a mask with a square region set to zero.
    """
    mask = tf.ones((image_size, image_size, 3), dtype=tf.float32)
    if fixed_position:
        x, y = fixed_position
    else:
        x = tf.random.uniform([], 0, image_size - mask_size + 1, dtype=tf.int32)
        y = tf.random.uniform([], 0, image_size - mask_size + 1, dtype=tf.int32)
    rows = tf.range(y, y + mask_size)
    cols = tf.range(x, x + mask_size)
    indices = tf.stack(tf.meshgrid(rows, cols, indexing='ij'), axis=-1)
    indices = tf.reshape(indices, (-1, 2))
    updates = tf.zeros((mask_size * mask_size, 3))
    mask = tf.tensor_scatter_nd_update(mask, indices, updates)
    return mask

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
