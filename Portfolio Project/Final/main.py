"""
Karl Estes
CSC 580 Portfolio Project
Created: March 28th, 2022
Due: April 10th, 2022

Asignment Prompt
----------------
The programming portion of the portfolio project required the implementation of a GAN which could learn to generate a set of images from the CIFAR 10 dataset.
The GAN architecture was provided per the assignment instructions.

File Description
----------------
Running this script will train a GAN on a random class from the CIFAR10 dataset. 
Samples of images from the generator will be saved every 10 epochs.
The generator model will also be saved every 10 epochs


Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import get_logger
from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm

# Disable everything but error message from Tensorflow
get_logger().setLevel('ERROR')

# Load CIFAR10
def load_dataset(class_num=None):

    (X, y), (_,_) = keras.datasets.cifar10.load_data()

    # Class number randomly chosen between 1 and 10
    if class_num is None:
        class_num = np.random.randint(low=0,high=10,size=1)[0]

    X=X[y.flatten() == class_num]
    X = X.astype('float32')

    X = (X / 127.5) - 1.0

    return X, class_num

# Function to select real samples from the dataset for training discriminator
def generate_real_samples(dataset, n_samples):
    # Choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)

    # Retrieve selected images
    X = dataset[ix]

    # Generate 'real' class labels (1)
    y = np.ones((n_samples, 1))

    return X, y

# Generate point in latent space (noise) as input for the generator
def generate_latent_points(latent_dim, n_samples):
    latent_points = np.random.randn(latent_dim * n_samples)

    # Reshape into batch of inputs for network
    latent_points = latent_points.reshape(n_samples, latent_dim)

    return latent_points

# Generate n fake samples with class labels
def generate_fake_samples(generator, latent_dim, n_samples):

    latent_points = generate_latent_points(latent_dim, n_samples)
    
    # Get output from generator
    X = generator.predict(latent_points)

    # Generate 'fake' class labels (0)
    y = np.zeros((n_samples, 1))

    return X, y

def build_discriminator(in_shape):

    model = Sequential()

    # normal
    model.add(Conv2D(32, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # downsample
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    #model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))

    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    #model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))

    # classifier
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002,0.5), metrics=['accuracy'])
    return model

def build_generator(latent_dim):

    # model = Sequential()

    # # foundation for 4x4 image
    # n_nodes = 256 * 4 * 4
    # model.add(Dense(n_nodes, input_dim=latent_dim))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Reshape((4, 4, 256)))

    # # upsample to 8x8
    # model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    # model.add(LeakyReLU(alpha=0.2))

    # # upsample to 16x16
    # model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    # model.add(LeakyReLU(alpha=0.2))

    # # upsample to 32x32
    # model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
    
    # # output layer
    # model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))

    model = Sequential()

    # Building the input layer
    model.add(Dense(128 * 8 * 8,activation='relu', input_dim=latent_dim))
    model.add(Reshape((8,8,128)))

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=3, padding="same"))
    #model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))

    model.add(UpSampling2D())

    model.add(Conv2D(64, kernel_size=3, padding="same"))
    #model.add(BatchNormalization(momentum=0.78))
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model

# Function for building combined GAN model
def build_gan(generator, discriminator):
    # Make discriminator untrainable so it doesn't update when training full GAN model
    discriminator.trainable = False # Does not impact training of standalone model

    model = Sequential()

    model.add(generator)
    model.add(discriminator)

    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002,0.5))
    return model

def save_plot(images, epoch, n=4):
    
    # Plot images
    for i in range(n*n):
        # Scale from [-1, 1] to [0, 1]
        image = (images[i] + 1) / 2.0

        plt.subplot(n,n,i+1)
        plt.axis('off')
        plt.imshow(image)
    
    # save plot to file
    filename = 'generated_plot_e{:03d}.jpg'.format(epoch)
    plt.savefig(os.path.join("plots", filename))
    plt.close()

def evaluate_performance(epoch, generator, discriminator, dataset, latent_dim, n_samples=200):

    # Evaluate discriminator on real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, real_acc = discriminator.evaluate(X_real, y_real, verbose=0)

    # Evaluate disciminator on fake smaples
    X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_samples)
    _, fake_acc = discriminator.evaluate(X_fake, y_fake, verbose=0)

    # Save plot with generated images
    save_plot(X_fake, epoch)

    # Save generator model
    filename = "generator_model_{:03d}.h5".format(epoch)
    generator.save(os.path.join("models", filename))

    return real_acc, fake_acc

def update_discriminator(discriminator, generator, dataset, latent_dim, n_iter=20, n_batch=128):
    half_batch = int(n_batch / 2)
    
    #for i in tqdm(range(n_iter), desc="Discriminator Update", position=2):
    for i in range(n_iter):
        
        # Update discriminator on real samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        d1_loss, real_acc = discriminator.train_on_batch(X_real, y_real)

        # Update disciminator on fake smaples
        X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
        d2_loss, fake_acc = discriminator.train_on_batch(X_fake, y_fake)

        return d1_loss, d2_loss
        #tqdm.write("Discriminator Update Epoch {}: real={}%% fake={}%%".format(i+1, real_acc*100, fake_acc*100))

# Function to train the discriminator
def train(generator, discriminator, gan, dataset, latent_dim, n_epochs=20, n_batch=128):
    batches_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch/2)

    #print("Trianing discriminating on valid images and random noise prior to GAN training...")
    #update_discriminator(discriminator, generator, dataset, latent_dim, 10, n_batch)

    for epoch in tqdm(range(n_epochs), desc="GAN Training", position=0):

        loss = []
        d1_loss = []
        d2_loss = []
            # update discriminator if either detection accuracy is less than 75%
            #if (real_acc < 0.75) or (fake_acc < 0.75):       

        for batch in range(batches_per_epoch):

            # Train discrimnator on real samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            dl1, _ = discriminator.train_on_batch(X_real, y_real)
            d1_loss.append(dl1)

            # Train discriminator on fake samples
            X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
            dl2, _ = discriminator.train_on_batch(X_fake, y_fake)
            d2_loss.append(dl2)

            # Prepare points in latent space as input for gan
            X_gan = generate_latent_points(latent_dim, n_batch)

            # Create inverted labels for fake samples
            y_gan = np.ones((n_batch, 1))

            # Update the generator via the discriminator's error
            g_loss = gan.train_on_batch(X_gan, y_gan)
            loss.append(g_loss)
            
            #tqdm.write("> {} ({}/{}): G={:.6f} | D ={:.6f} - D2={:.6f}".format(epoch+1, batch+1, batches_per_epoch, g_loss, dl1, dl2))

        # Summarize average loss across epoch
        tqdm.write("*> Epoch {}: Generator Loss={:.6f} | Discriminator Loss = {:.6f} - {:.6f}".format(epoch+1, np.average(loss),np.average(d1_loss), np.average(d2_loss)))

        #update_discriminator(discriminator, generator, dataset, latent_dim, 10, n_batch)

        if ((epoch+1) % 10 == 0) or (epoch == 0):
            real_acc, fake_acc = evaluate_performance(epoch+1, generator, discriminator, dataset, latent_dim, 100)
            tqdm.write("Evaluation > Epoch {}: real: {}%, fake: {}%".format(epoch+1, real_acc*100, fake_acc*100))

            if (real_acc < 0.8) or (fake_acc < 0.8):
              tqdm.write("** Discriminator accuracy below threshold of 80% - Doing some additional Discriminator updating **")
              update_discriminator(discriminator, generator, dataset, latent_dim, 40, 512)
              
if __name__ == "__main__":
    image_shape = (32, 32, 3)
    latent_dimensions = 100
    dataset,_ = load_dataset()

    discriminator_model = build_discriminator(image_shape)
    generator_model = build_generator(latent_dimensions)
    gan_model = build_gan(generator_model, discriminator_model)

    train(generator_model, discriminator_model, gan_model, dataset, latent_dimensions, 200, 128)