"""
Karl Estes
CSC 580 Critical Thinking 6

Script which loads the CIFAR10 dataset, does a little hyperparameter tuning on a few CNN architecures, train the best models on the CIFAR10 set for 10 epochs, then tests
each model and outputs the accuracy

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""


import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models, layers
from threading import Thread
from time import sleep
from progressbar import ProgressBar, UnknownLength
from tqdm.keras import TqdmCallback

# Model from Tensorflow Tutorial
def build_model_tf():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10)) # Ten total output classes

    # Compile model
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

def build_model_pool(hp):
    
    # Tune the number of conv layers
    model = models.Sequential(name="Conv_Pooling")
    model.add(layers.Input((32,32,3)))

    for i in range(hp.Int("num_conv_layers", 1, 3)):
        
        # Add Conv Layer
        model.add(
            layers.Conv2D(
                # Tune number of units
                hp.Int(f"conv_units_{i}", min_value=32, max_value=128,step=32),
                # Tune kernel size
                kernel_size=hp.Choice(f"kernel_size_{i}", [3,5]),
                activation='relu',
                padding="same"
            ))

        # Add Pooling Layer
        model.add(
            layers.MaxPooling2D((2,2))
        )

    model.add(layers.Flatten())

    if(hp.Boolean("dropout")):
        model.add(layers.Dropout(rate=0.5))

    # Tune number of conv layers
    for i in range(hp.Int("num_dense_layers", 1, 3)):
        model.add(
            layers.Dense(units=hp.Int(f"dense_units_{i}", min_value=32, max_value=256, step=32), activation='relu')
        )

    model.add(layers.Dense(10))

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

# Tuned model w/o pooling
def build_model_no_pool(hp):
    
    # Tune the number of conv layers
    model = models.Sequential(name="Conv_No_Pooling")
    model.add(layers.Input((32,32,3)))

    for i in range(hp.Int("num_conv_layers", 1, 3)):
        
        # Add Conv Layer
        model.add(
            layers.Conv2D(
                # Tune number of units
                hp.Int(f"conv_units_{i}", min_value=32, max_value=128,step=32),
                # Tune kernel size
                kernel_size=hp.Choice(f"kernel_size_{i}", [3,5]),
                activation='relu',
                padding="same"
            ))

    model.add(layers.Flatten())

    if(hp.Boolean("dropout")):
        model.add(layers.Dropout(rate=0.5))

    # Tune number of conv layers
    for i in range(hp.Int("num_dense_layers", 1, 3)):
        model.add(
            layers.Dense(units=hp.Int(f"dense_units_{i}", min_value=32, max_value=256, step=32), activation='relu')
        )

    model.add(layers.Dense(10))

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

# Tuned dilated model
def build_model_dilation(hp):
    
    # Tune the number of conv layers
    model = models.Sequential(name="Conv_Dilation")
    model.add(layers.Input((32,32,3)))

    for i in range(hp.Int("num_conv_layers", 1, 3)):
        
        # Add Conv Layer
        model.add(
            layers.Conv2D(
                # Tune number of units
                hp.Int(f"conv_units_{i}", min_value=32, max_value=128,step=32),
                # Tune kernel size
                kernel_size=hp.Choice(f"kernel_size_{i}", [3,5]),
                # Tune strides
                dilation_rate=hp.Int(f"dilation_{i}", min_value=1, max_value=3),
                activation='relu',
                padding="same"
            ))

            # Add Pooling Layer
        model.add(
            layers.MaxPooling2D((2,2))
        )

    model.add(layers.Flatten())

    if(hp.Boolean("dropout")):
        model.add(layers.Dropout(rate=0.5))

    # Tune number of conv layers
    for i in range(hp.Int("num_dense_layers", 1, 3)):
        model.add(
            layers.Dense(units=hp.Int(f"dense_units_{i}", min_value=32, max_value=256, step=32), activation='relu')
        )

    model.add(layers.Dense(10))

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

def build_model_dilation_no_pool(hp):
    
    # Tune the number of conv layers
    model = models.Sequential(name="Conv_Dilation_No_Pooling")
    model.add(layers.Input((32,32,3)))

    for i in range(hp.Int("num_conv_layers", 1, 3)):
        
        # Add Conv Layer
        model.add(
            layers.Conv2D(
                # Tune number of units
                hp.Int(f"conv_units_{i}", min_value=32, max_value=128,step=32),
                # Tune kernel size
                kernel_size=hp.Choice(f"kernel_size_{i}", [3,5]),
                # Tune strides
                dilation_rate=hp.Int(f"dilation_{i}", min_value=1, max_value=3),
                activation='relu',
                padding="same"
            ))

    model.add(layers.Flatten())

    if(hp.Boolean("dropout")):
        model.add(layers.Dropout(rate=0.5))

    # Tune number of conv layers
    for i in range(hp.Int("num_dense_layers", 1, 3)):
        model.add(
            layers.Dense(units=hp.Int(f"dense_units_{i}", min_value=32, max_value=256, step=32), activation='relu')
        )

    model.add(layers.Dense(10))

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

def load_data():
    # Download the CIFAR10 Dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    # Normalize the data
    train_images, test_images = train_images/255.0, test_images/255.0

    return train_images, train_labels, test_images, test_labels

def hpo_helper(tuner, X, y, epochs, batch_size, verbose):
    tuner.search(X,y,epochs=epochs, batch_size=batch_size, verbose=verbose)

def hpo(max_trials, executions_per_trials, X, y, epochs, batch_size, verbose, num_models):

    # Define four tuners

    tuner_pool = kt.BayesianOptimization(
        hypermodel=build_model_pool,
        objective='accuracy',
        max_trials=max_trials,
        executions_per_trial= executions_per_trials,
        overwrite=True,
        directory="hpo/pool",
        project_name="CIFAR10"
    )

    tuner_no_pool = kt.BayesianOptimization(
        hypermodel=build_model_no_pool,
        objective='accuracy',
        max_trials=max_trials,
        executions_per_trial= executions_per_trials,
        overwrite=True,
        directory="hpo/no_pool",
        project_name="CIFAR10"
    )

    tuner_dilation = kt.BayesianOptimization(
        hypermodel=build_model_dilation,
        objective='accuracy',
        max_trials=max_trials,
        executions_per_trial= executions_per_trials,
        overwrite=True,
        directory="hpo/dilation",
        project_name="CIFAR10"
    )

    tuner_dilation_no_pool = kt.BayesianOptimization(
        hypermodel=build_model_dilation_no_pool,
        objective='accuracy',
        max_trials=max_trials,
        executions_per_trial= executions_per_trials,
        overwrite=True,
        directory="hpo/dilation_no_pool",
        project_name="CIFAR10"
    )

    # Spawn threads and wait
    bar = ProgressBar(max_value=UnknownLength)

    workers = [
        Thread(target=hpo_helper, args=(tuner_pool, X, y, epochs, batch_size, verbose)),
        Thread(target=hpo_helper, args=(tuner_no_pool, X, y, epochs, batch_size, verbose)),
        Thread(target=hpo_helper, args=(tuner_dilation, X, y, epochs, batch_size, verbose)),
        Thread(target=hpo_helper, args=(tuner_dilation_no_pool, X, y, epochs, batch_size, verbose))
    ]

    # Start threads
    for th in workers:
        th.start()

    alive = True

    while(alive):
        alive = False
        for th in workers:
            if th.is_alive():
                alive = True
        bar.update()
        sleep(1)

    # Ensure threads have joined up
    for th in workers:
        th.join()

    return tuner_pool.get_best_models(num_models=num_models), tuner_no_pool.get_best_models(num_models=num_models), tuner_dilation.get_best_models(num_models=num_models), tuner_dilation_no_pool.get_best_models(num_models=num_models)

if __name__ == "__main__":

    print("Loading dataset...")
    train_X, train_y, test_X, test_y = load_data()

    print("HPO...")
    pool, no_pool, dilated, dilated_no_pool = hpo(4, 1, train_X, train_y, 1, 200, 0, 1)

    model_basic = build_model_tf()
    model_pool = pool[0]
    model_no_pool = no_pool[0]
    model_dilated = dilated[0]
    model_dilated_no_pool = dilated_no_pool[0]

    # TODO - FIT THE MODELS
    print("\n\nFitting models\n")

    hist_basic = model_basic.fit(train_X, train_y, epochs=10, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    hist_pool = model_pool.fit(train_X, train_y, epochs=10, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    hist_no_pool = model_no_pool.fit(train_X, train_y, epochs=10, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    hist_dilated = model_dilated.fit(train_X, train_y, epochs=10, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    hist_dilated_no_pool = model_dilated_no_pool.fit(train_X, train_y, epochs=10, verbose=0, callbacks=[TqdmCallback(verbose=2)])

    # TODO - TEST THE MODELS
    test_loss, test_acc = model_basic.evaluate(test_X, test_y, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    print("Model: {} | Loss: {} | Accuracy | {}".format("Basic", test_loss, test_acc))
    test_loss, test_acc = model_pool.evaluate(test_X, test_y, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    print("Model: {} | Loss: {} | Accuracy | {}".format("Pooling", test_loss, test_acc))
    test_loss, test_acc = model_no_pool.evaluate(test_X, test_y, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    print("Model: {} | Loss: {} | Accuracy | {}".format("No Pooling", test_loss, test_acc))
    test_loss, test_acc = model_dilated.evaluate(test_X, test_y, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    print("Model: {} | Loss: {} | Accuracy | {}".format("Dilation", test_loss, test_acc))
    test_loss, test_acc = model_dilated_no_pool.evaluate(test_X, test_y, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    print("Model: {} | Loss: {} | Accuracy | {}".format("Dilation No Pooling", test_loss, test_acc))

    # Plot Graphs with MatplotLib
    for history in [hist_basic, hist_pool, hist_no_pool, hist_dilated, hist_dilated_no_pool]:
        plt.plot(history.history['loss'], label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()