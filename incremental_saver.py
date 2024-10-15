import os

from Constants import MODELS_DIR


def get_last_models():
    """
    Get the last saved models from the models directory that is saved during training.
    """
    generators = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if 'generator' in f]
    discriminators = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if 'discriminator' in f]
    if len(generators) == 0 or len(discriminators) == 0:
        return None, None
    last_generator = sorted(generators)[-1]
    last_discriminator = sorted(discriminators)[-1]
    return last_generator, last_discriminator


def get_model_name_to_save():
    """
    Get the name of the model to save based on the number of models saved in the models directory.
    """
    generators = [f for f in os.listdir(MODELS_DIR) if 'generator' in f]
    discriminators = [f for f in os.listdir(MODELS_DIR) if 'discriminator' in f]
    return f'generator_{len(generators)}.keras', f'discriminator_{len(discriminators)}.keras'


def save_models(generator, discriminator):
    """
    Save the generator and discriminator models to the models directory.
    """
    generator_name, discriminator_name = get_model_name_to_save()
    generator.save(os.path.join(MODELS_DIR, generator_name))
    discriminator.save(os.path.join(MODELS_DIR, discriminator_name))
    print("Models saved successfully.")
