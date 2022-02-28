"""
Karl Estes
CSC 580 Critical Thinking 2 - Predicting Future Sales
Created: Feburary 24th, 2021
Due: Feburary 27th, 2021

Asignment Prompt
----------------
In a nutshell, *sales_data_test.csv* and *sales_data_test.csv* contain data that will be used to train a neural network to predict how much money can be expected 
from the future sale of new video games. The .csv files were retrieved from one of [Toni Esteves repos](https://github.com/toniesteves/adam-geitgey-building-deep-learning-keras/tree/master/03). 

The columns in the data are defined as follows:
- critic_rating : an average rating out of five stars
- is_action : tells us if this was an action game
- is_exclusive_to_us : tells us if we have an exclusiv deal to sell this game
- is_portable : tells us if this game runs on a handheld video game system
- is_role_playing : tells us if this is a role-playing game
- is_sequel : tells us if this game was a sequel to an earlier video game and part of an ongoing series
- is_sports : tells us if this was a sports game
- suitable_for_kids : tells us if this game is appropriate for all ages
- total_earning : tells us how much money the store has earned in total from selling the game to all customers
- unit_price : tells us for how much a single copy of the game retailed

File Description
----------------
This script produces two different models for predicting future sales of a video game. Both models are constructed with Keras and are Sequential models. One model 
was constructed via the assignment parameters and was constructed of a few Dense hidden layers with a single output node. The other model was 'optimized' using the 
keras_tuner library. 

All the data preprocessing and outputs are structured as per the assignment parameters. The inclusion of hyperparemter tuning via the keras_tuner library
was a deviation from the assignment instructions. The core assingment was still completed, however, and this gave me a chance to exlplore hyperparemter tuning.

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

from distutils.command.build import build
from tabnanny import verbose
from matplotlib.pyplot import bar
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from keras.models import Sequential
from keras import layers 
from keras import activations
import argparse

# Global
barRunning = True;

def prep_data():
    # Load the training and testing data
    train_data = pd.read_csv("sales_data_training.csv", dtype=float)
    test_data = pd.read_csv("sales_data_test.csv", dtype = float)

    # Scale the data using sklearn
    scaler = MinMaxScaler(feature_range=(0,1))
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.fit_transform(test_data)

    # Print out adjustment
    print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

    # Create new DataFrames
    df_train_scaled = pd.DataFrame(train_data_scaled, columns=train_data.columns.values)
    df_test_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns.values)

    # Save scaled data
    df_train_scaled.to_csv("sales_data_training_scaled.csv", index=False)
    df_test_scaled.to_csv("sales_data_testing_scaled.csv", index=False)

    return scaler

def load_training_data():

    # Load the training data
    training_data_df = pd.read_csv("sales_data_training_scaled.csv")

    X = training_data_df.drop('total_earnings', axis=1).values
    Y = training_data_df[['total_earnings']].values

    return X,Y

def build_model():
    model = Sequential()
    model.add(layers.Input((9,)))
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(32, activation=activations.relu))
    model.add(layers.Dense(1, activation=activations.linear))
    model.compile('adam', loss='mse')

    return model

def build_model_tuning(hp):
    model = Sequential()
    model.add(layers.Input((9,))) # Input

    # Tune the number of layers
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # Tune number of units separately
                units=hp.Int("units_{i}", min_value=32, max_value=512, step=32),
                activation=activations.relu
            )
        )
    model.add(layers.Dense(1, activation=activations.linear)) # Output

    # Tune the learning rate
    learning_rate = hp.Float("lr", min_value = 1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=hp.Choice("loss", ['mse','mae']),
        metrics=['mse',tf.keras.metrics.RootMeanSquaredError(),'mae'])

    return model

def tune_hyperparameters(X,Y,val_size: float = 0.2, verbose: int = 2):
    global barRunning
    print("\n* * * * * Hyperparemter Tuning Under Way* * * * *")

    # Tuning the model's hyperparameters
    tuner = kt.BayesianOptimization(
        hypermodel=build_model_tuning,
        objective=kt.Objective("val_root_mean_squared_error", direction="min"),
        max_trials=15,
        executions_per_trial=2,
        overwrite=True,
        directory="hpo",
        project_name="sales_predictions"
    )

    # Summary of the search space
    tuner.search_space_summary()


    X_TRAIN,X_VAL,Y_TRAIN,Y_VAL = train_test_split(X,Y,test_size=val_size)
    # Search for the best hyperparemeter configurations
    tuner.search(X_TRAIN, Y_TRAIN, epochs=10, batch_size=25, validation_data=(X_VAL, Y_VAL), verbose=verbose)

    print("\n* * * * * Best Tuned Model * * * * *")
    # Get the best model and return it
    models = tuner.get_best_models(num_models=1)
    best_model = models[0]

    best_model.build()
    best_model.summary()


    return best_model

def train_model(X,Y,model,name: str, verbose: int = 2, validation_split: float = 0.2):

    print("\n* * * * * Training Model * * * * *")
    model.fit(X,Y,batch_size = 100, epochs = 50, verbose=verbose, shuffle=True, validation_split=validation_split)

    print("\n* * * * * Saving Model: {} * * * * *".format(name))
    model.save("{}.h5".format(name))

    return model

def test_model(model):
    # Load the testing data
    testing_data_df = pd.read_csv("sales_data_testing_scaled.csv")

    X_TEST = testing_data_df.drop('total_earnings', axis=1).values
    Y_TEST = testing_data_df[['total_earnings']].values

    test_error_rate = model.evaluate(X_TEST, Y_TEST, verbose=2)

    if type(test_error_rate) is float:
        error = test_error_rate
    else:
        error = test_error_rate[1]
        
    print("The mean squared error (MSE) for the test data set is: {}".format(error))

def predict(model, scaler):

    X_PRED = pd.read_csv("proposed_new_product.csv").values
    pred = model.predict(X_PRED)

    temp = np.concatenate((X_PRED[0,0:8],pred[0], [X_PRED[0][8]]))
    scaled = scaler.inverse_transform([temp])

    # Scale
    #pred = pred[0][0]
    #pred_hp = pred_hp[0][0]
    #pred = (pred + scaler.min_[8]) / scaler.scale_[8]
    #pred_hp = (pred_hp + scaler.min_[8]) / scaler.scale_[8]
    #print("Earnings Prediction for Proposed Product - (Non-HP): ${}  |  (HP): ${}".format(pred, pred_hp))

    print("Earnings Prediction for Proposed Product: ${}".format(scaled[0][8]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo", help="When flag is passed, will create a model and tune various hyperparemters rather than using the harcoded model", action="store_true")

    args = parser.parse_args()

    # Data preprocessing
    scaler = prep_data()

    # Load data
    X_train, Y_train = load_training_data()

    if(args.hpo): # Run with hyperparetmer tuning steps
        #model = build_model_tuning()
        tuned_model = tune_hyperparameters(X_train, Y_train,verbose=1)
        trained_model = train_model(X_train, Y_train, tuned_model, "trained_tuned_model", verbose=2, validation_split=0)
    else: # Run without tuning
        model = build_model()
        trained_model = train_model(X_train, Y_train, model, "trained_untuned_model", verbose=2, validation_split=0)

    test_model(trained_model)

    predict(trained_model, scaler)
