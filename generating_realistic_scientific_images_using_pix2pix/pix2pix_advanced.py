import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime

# Ensure TensorFlow 2.x compatibility
tf.keras.backend.set_floatx('float32')

# Define hyperparameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.0002
BETA_1 = 0.5
LAMBDA_L1 = 100.0  # Weight for L1 loss as per Pix2Pix paper
PATCH_SIZE = 30    # Output patch size for 70x70 PatchGAN discriminator

# Define directories for saving models and plots
SAVE_DIR = "pix2pix_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < EPOCHS // 2:
        return lr
    else:
        return lr * tf.math.exp(-0.1)  # Exponential decay after half of epochs

# Data preprocessing function
def preprocess_image(image):
    """Normalize images to [-1, 1] as required by the generator's tanh output."""
    return (image / 127.5) - 1.0

def deprocess_image(image):
    """Convert images from [-1, 1] to [0, 1] for visualization."""
    return (image + 1.0) / 2.0

# Data augmentation function
def augment_image(src, tgt):
    """Apply random jittering and mirroring."""
    # Random jitter: resize to 286x286 and random crop back to 256x256
    src = tf.image.resize(src, [286, 286])
    tgt = tf.image.resize(tgt, [286, 286])
    src = tf.image.random_crop(src, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    tgt = tf.image.random_crop(tgt, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    
    # Random mirroring
    if tf.random.uniform(()) > 0.5:
        src = tf.image.flip_left_right(src)
        tgt = tf.image.flip_left_right(tgt)
    
    return src, tgt

# Load and prepare dataset (example placeholder, replace with actual dataset loading)
def load_dataset():
    """
    Placeholder for dataset loading. Replace with actual dataset loading logic.
    Expected format: (trainA, trainB) where trainA and trainB are numpy arrays of shape (N, H, W, C).
    """
    # Example: Load Cityscapes or other paired dataset
    # For demonstration, create dummy data
    trainA = np.random.rand(100, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    trainB = np.random.rand(100, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    
    # Preprocess images
    trainA = preprocess_image(trainA)
    trainB = preprocess_image(trainB)
    
    return trainA, trainB

# Define the discriminator (PatchGAN)
def define_discriminator(image_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """Define a PatchGAN discriminator as per Pix2Pix paper."""
    init = RandomNormal(stddev=0.02)
    
    # Inputs: source and target images
    in_src_image = Input(shape=image_shape, name='source_image')
    in_target_image = Input(shape=image_shape, name='target_image')
    
    # Concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    
    # C64: 4x4 kernel, stride 2
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    # Second last layer: 4x4 kernel, stride 1
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    # Patch output: 4x4 kernel, stride 1, sigmoid activation
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    
    # Define model
    model = Model([in_src_image, in_target_image], patch_out, name='discriminator')
    
    # Compile model
    opt = Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    
    return model

# Define encoder block for generator
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    """Define an encoder block for the U-Net generator."""
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g

# Define decoder block for generator
def define_decoder_block(layer_in, skip_in, n_filters, dropout=True):
    """Define a decoder block for the U-Net generator."""
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g

# Define the generator (U-Net)
def define_generator(image_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """Define a U-Net generator as per Pix2Pix paper."""
    init = RandomNormal(stddev=0.02)
    
    # Image input
    in_image = Input(shape=image_shape, name='input_image')
    
    # Encoder: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    
    # Bottleneck
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    
    # Decoder: CD512-CD512-CD512-C512-C256-C128-C64
    d1 = define_decoder_block(b, e7, 512)
    d2 = define_decoder_block(d1, e6, 512)
    d3 = define_decoder_block(d2, e5, 512)
    d4 = define_decoder_block(d3, e4, 512, dropout=False)
    d5 = define_decoder_block(d4, e3, 256, dropout=False)
    d6 = define_decoder_block(d5, e2, 128, dropout=False)
    d7 = define_decoder_block(d6, e1, 64, dropout=False)
    
    # Output
    g = Conv2DTranspose(IMG_CHANNELS, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)  # Output in [-1, 1]
    
    # Define model
    model = Model(in_image, out_image, name='generator')
    return model

# Define the combined GAN model
def define_gan(g_model, d_model, image_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """Define the combined GAN model for training the generator."""
    # Make discriminator weights non-trainable
    d_model.trainable = False
    
    # Define source image input
    in_src = Input(shape=image_shape, name='source_image')
    
    # Generator output
    gen_out = g_model(in_src)
    
    # Discriminator output
    dis_out = d_model([in_src, gen_out])
    
    # Define model: input is source image, outputs are discriminator output and generated image
    model = Model(in_src, [dis_out, gen_out], name='gan')
    
    # Compile model
    opt = Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
    model.compile(loss=['binary_crossentropy', 'mae'], 
                  optimizer=opt, 
                  loss_weights=[1, LAMBDA_L1])
    
    return model

# Generate real samples
@tf.function
def generate_real_samples(dataset, n_samples, patch_shape):
    """Select a batch of real samples."""
    trainA, trainB = dataset
    ix = tf.random.uniform((n_samples,), 0, trainA.shape[0], dtype=tf.int32)
    X1, X2 = tf.gather(trainA, ix), tf.gather(trainB, ix)
    
    # Apply augmentation
    X1, X2 = augment_image(X1, X2)
    
    # Generate 'real' class labels (1)
    y = tf.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

# Generate fake samples
@tf.function
def generate_fake_samples(g_model, samples, patch_shape):
    """Generate a batch of fake samples using the generator."""
    X = g_model(samples)
    y = tf.zeros((tf.shape(X)[0], patch_shape, patch_shape, 1))
    return X, y

# Summarize performance and save models/plots
def summarize_performance(step, g_model, dataset, n_samples=3):
    """Generate and save sample images and model checkpoints."""
    # Select real samples
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    
    # Generate fake samples
    X_fakeB = g_model.predict(X_realA)
    
    # Deprocess images for visualization
    X_realA = deprocess_image(X_realA)
    X_realB = deprocess_image(X_realB)
    X_fakeB = deprocess_image(X_fakeB)
    
    # Plot images
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        # Plot source images
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.title('Source')
        plt.imshow(X_realA[i])
        
        # Plot generated images
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.title('Generated')
        plt.imshow(X_fakeB[i])
        
        # Plot target images
        plt.subplot(3, n_samples, 1 + 2 * n_samples + i)
        plt.axis('off')
        plt.title('Target')
        plt.imshow(X_realB[i])
    
    # Save plot
    filename1 = os.path.join(SAVE_DIR, f'plot_{step+1:06d}.png')
    plt.savefig(filename1)
    plt.close()
    
    # Save model
    filename2 = os.path.join(SAVE_DIR, f'model_{step+1:06d}.h5')
    g_model.save(filename2)
    
    print(f'>Saved: {filename1} and {filename2}')

# Train the Pix2Pix model
def train(d_model, g_model, gan_model, dataset, n_epochs=EPOCHS, n_batch=BATCH_SIZE):
    """Train the Pix2Pix GAN model."""
    # Calculate patch shape for discriminator output
    n_patch = d_model.output_shape[1]
    
    # Unpack dataset
    trainA, trainB = dataset
    
    # Calculate batches per epoch
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    
    # Define callbacks
    lr_callback = LearningRateScheduler(lr_scheduler)
    
    # Training loop
    for i in range(n_steps):
        # Select real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        
        # Generate fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        
        # Update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        
        # Update discriminator for fake samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        
        # Update generator via GAN model
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        
        # Log losses
        print(f'>{i+1}/{n_steps}, d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}')
        
        # Summarize performance periodically
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)
        
        # Apply learning rate schedule
        lr_callback.on_epoch_end(i // bat_per_epo, gan_model.optimizer)
        lr_callback.on_epoch_end(i // bat_per_epo, d_model.optimizer)

# Main execution
if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset()
    
    # Define models
    d_model = define_discriminator()
    g_model = define_generator()
    gan_model = define_gan(g_model, d_model)
    
    # Print model summaries
    d_model.summary()
    g_model.summary()
    gan_model.summary()
    
    # Train the model
    train(d_model, g_model, gan_model, dataset)