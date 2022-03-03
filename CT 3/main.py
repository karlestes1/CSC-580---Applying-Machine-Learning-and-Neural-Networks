"""
Karl Estes
CSC 580 Critical Thinking 3 - Predicting Fuel Efficiency Using TensorFlow
Created: February 28th, 2022
Due: March 6th, 2022

Asignment Prompt
----------------
In a nutshell, this is a *regression* problem where a **neural network** will be created with the ```tf.keras``` API and will utilize the 
[**Auto MPG**](https://archive.ics.uci.edu/ml/datasets/auto+mpg) dataset. The trained model will be used to predict the feul efficiency of 
late 1970s and early 1980s automobiles

File Description
----------------
TODO - File Description Here

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

def data_preprocessing():
    """
    Retrieves the Auto MPG dataset, creates a pandas dataframe, and normalizes the training
    and testing data. 

    Information on the train_dataset is presented in the form of a pairplot as per assignment instruction.
    
    Returns: X_train, Y_train, X_test, Y_test
    """

    print("\n* * * * * DATA PREPROCESSING * * * * *")

    # Download the dataset
    data_file = keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    # Import data into Pandas
    column_names =["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]
    raw_data = pd.read_csv(data_file, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
    dataset=raw_data.copy()

    print("\n* * Dataset Tail * *")
    print(dataset.tail())


    print("\n* * Splitting Train/Test data * *")
    # Train/test dataset split
    train_dataset = dataset.sample(frac=0.8, random_state=0) 
    test_dataset = dataset.drop(train_dataset.index)

    # Had to include because some missing values made it through. Fixes an issue of getting loss and metrics with 'nan' value
    train_dataset.dropna(inplace=True)
    test_dataset.dropna(inplace=True)

    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    plt.suptitle("Train_Dataset Visualization (Subset of Variables)")
    plt.show()

    print("\n* * Training Data Info * *")
    # Info on training data
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    train_stats

    print("\n* * Normalizing Data * *")
    def norm(x):
        return (x-train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    # Separate features and labels
    X_train, Y_train = normed_train_data.drop("MPG", axis=1), train_dataset["MPG"]
    X_test, Y_test = normed_test_data.drop("MPG", axis=1), test_dataset["MPG"]

    return X_train, Y_train, X_test, Y_test

def build_model(loss: str = 'mse', input_len: int = 9):
    model = keras.Sequential(name=loss)
    model.add(keras.layers.Input(input_len))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae','mse'])

    return model

if __name__ == "__main__":

    EPOCHS = 1000

    # Get Data
    X_train, Y_train, X_test, Y_test = data_preprocessing()

    print("\n* * * * * Constructing Models * * * * *")
    # Construct models - One with MSE loss and one with MAE loss
    model_mse = build_model('mse', len(X_train.keys()))
    model_mae = build_model('mae', len(X_train.keys()))

    model_mse.summary()
    model_mae.summary()

    print("\n* * * * * Running Prediction Test (Required by Assignment Instructions) * * * * *")
    print(model_mse.predict(X_train[0:10]))
    print(model_mae.predict(X_train[0:10]))

    print("\n* * * * * Training the Models * * * * *")
    print("* * * MSE Model * * *")
    history = model_mse.fit(X_train, Y_train, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print("\n")
    print(hist.tail())
   
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mae")  
    plt.ylim([0, 10])  
    plt.ylabel('MAE [MPG]')
    plt.show()
    plotter.plot({'Basic': history}, metric = "mse")  
    plt.ylim([0, 20])  
    plt.ylabel('MSE [MPG^2]')   
    plt.show()

    print("* * * MAE Model * * *")
    history = model_mae.fit(X_train, Y_train, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print("\n")
    print(hist.tail())

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mae")  
    plt.ylim([0, 10])  
    plt.ylabel('MAE [MPG]')
    plt.show()
    plotter.plot({'Basic': history}, metric = "mse")  
    plt.ylim([0, 20])  
    plt.ylabel('MSE [MPG^2]')   
    plt.show()