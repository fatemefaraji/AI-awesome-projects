import numpy as np
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Dropout, BatchNormalization, LeakyReLU, Input, Reshape
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from numpy.random import randint, randn, choice

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Preparation
def get_images_ready(n_classes=10):
    (trainX, trainy), (testX, testy) = load_data()
    # Expand dimensions and normalize
    X = np.expand_dims(trainX, axis=-1).astype('float32') / 255.0
    testX = np.expand_dims(testX, axis=-1).astype('float32') / 255.0
    print(f"Train shape: {X.shape}, Test shape: {testX.shape}")
    return (X, trainy), (testX, testy)

def select_subset_images(dataset, n_samples=120, n_classes=10):
    X, y = dataset
    X_list, y_list = [], []
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):
        X_with_class = X[y == i]
        ix = randint(0, len(X_with_class), n_per_class)
        X_list.extend(X_with_class[ix])
        y_list.extend([i] * n_per_class)
    return np.array(X_list), np.array(y_list)

# Load and prepare data
(trainX, trainy), (testX, testy) = get_images_ready()
X_labeled, y_labeled = select_subset_images((trainX, trainy))
X_unlabeled = trainX  # Use all training data as unlabeled pool
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

# 2. Define GAN Models
def define_discriminator(n_classes=10):
    input_shape = (28, 28, 1)
    model = Sequential([
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dropout(0.4),
        
        Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dropout(0.4),
        
        Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        
        Flatten(),
        Dense(n_classes + 1, activation='softmax')  # n_classes + 1 for "fake" class
    ])
    return model

def define_generator(latent_dim=100):
    model = Sequential([
        Dense(7 * 7 * 256, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Reshape((7, 7, 256)),
        BatchNormalization(),
        
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        
        Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        
        Conv2D(1, (3, 3), padding='same', activation='sigmoid')
    ])
    return model

def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    return model

# Instantiate models
latent_dim = 100
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan = define_gan(generator, discriminator)

# Compile models
opt_disc = Adam(lr=0.0002, beta_1=0.5, clipvalue=1.0)
opt_gan = Adam(lr=0.0002, beta_1=0.5, clipvalue=1.0)
discriminator.compile(optimizer=opt_disc, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=opt_gan, loss='binary_crossentropy')

# 3. Training Functions
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input, verbose=0)
    y = np.ones((n_samples, 1)) * 10  # Class 10 for fake
    return X, y

def train_gan(generator, discriminator, gan, labeled_data, unlabeled_data, n_epochs=100, batch_size=64, latent_dim=100):
    X_labeled, y_labeled = labeled_data
    X_unlabeled = unlabeled_data
    n_classes = 10
    bat_per_epo = int(len(X_labeled) / batch_size)
    
    history = {'d_loss': [], 'd_acc': [], 'g_loss': [], 'val_acc': []}
    
    for epoch in range(n_epochs):
        d_loss, d_acc, g_loss = 0, 0, 0
        for _ in range(bat_per_epo):
            # Train on labeled data
            idx = choice(len(X_labeled), batch_size // 3)
            X_real_labeled, y_real_labeled = X_labeled[idx], y_labeled[idx]
            
            # Train on unlabeled data
            idx = choice(len(X_unlabeled), batch_size // 3)
            X_real_unlabeled = X_unlabeled[idx]
            y_real_unlabeled = np.random.randint(0, n_classes, (batch_size // 3, 1))  # Pseudo-labels
            
            # Train on fake data
            X_fake, y_fake = generate_fake_samples(generator, latent_dim, batch_size // 3)
            
            # Combine data
            X_disc = np.vstack([X_real_labeled, X_real_unlabeled, X_fake])
            y_disc = np.vstack([y_real_labeled.reshape(-1, 1), y_real_unlabeled, y_fake])
            
            # Train discriminator
            d_metrics = discriminator.train_on_batch(X_disc, y_disc)
            d_loss += d_metrics[0]
            d_acc += d_metrics[1]
            
            # Train generator
            X_gan = generate_latent_points(latent_dim, batch_size)
            y_gan = np.zeros((batch_size, 1))  # Trick discriminator into thinking fake is real
            g_loss += gan.train_on_batch(X_gan, y_gan)
        
        # Average metrics
        d_loss /= bat_per_epo
        d_acc /= bat_per_epo
        g_loss /= bat_per_epo
        
        # Validation accuracy
        _, val_acc = discriminator.evaluate(X_val, y_val, verbose=0)
        
        # Log metrics
        history['d_loss'].append(d_loss)
        history['d_acc'].append(d_acc)
        history['g_loss'].append(g_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{n_epochs} - D Loss: {d_loss:.4f}, D Acc: {d_acc:.4f}, G Loss: {g_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Visualize generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            X_fake, _ = generate_fake_samples(generator, latent_dim, 16)
            plt.figure(figsize=(4, 4))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(X_fake[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.show()
    
    return history

# 4. Train the GAN
history = train_gan(generator, discriminator, gan, (X_train, y_train), X_unlabeled, n_epochs=100, batch_size=64, latent_dim=latent_dim)

# 5. Evaluate on Test Set
_, test_acc = discriminator.evaluate(testX, testy, verbose=0)
print(f'Test Accuracy: {test_acc * 100:.3f}%')

# Predictions and Confusion Matrix
y_pred = discriminator.predict(testX)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(testy, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 6. Plot Training Metrics
epochs = range(1, len(history['d_loss']) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history['d_loss'], label='Discriminator Loss')
plt.plot(epochs, history['g_loss'], label='Generator Loss')
plt.title('Training Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history['d_acc'], label='Discriminator Accuracy')
plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 7. Save Models
discriminator.save('ssgan_discriminator.h5')
generator.save('ssgan_generator.h5')