"""
Karl Estes
CSC 580 Critical Thinking 4 - Toxicology Testing
Created: March 10th, 2022
Due: March 13th, 2022

Asignment Prompt
----------------
For this assignment, you will use a chemical dataset. 
Toxicologists are very interested in the task of using machine learning to predict whether a given compound will be toxic.
This task is extremely complicated because science has only a limited understanding of the metabolic processes that happen in a human body.
Biologists and chemists, however, have worked out a limited set of experiments that provide indications of toxicity. 
If a compound it a "hit" in one of these experiments, it will likely be toxic for humans to ingest.

Some sample code was provided for the creation of a neural network in TensorFlow V1. The assignment required adding dropout to the predifined network,
calculating accuracy on the validation set, and logging results and graph structure via Tensorboard

File Description
----------------
TODO - Put something here

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from deepchem import deepchem as dc
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Seeds are set via assignment parameters
np.random.seed(456)
tf.random.set_seed(456)

# Disables eager execution so TF v1 code can be run
tf.compat.v1.disable_eager_execution()


# Arguments for script
parser = argparse.ArgumentParser()
parser.add_argument("--n_hidden", type=int, default=50, help="Number of neurons in hidden layer")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (Using AdamOptimizer)")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of trianing epochs")
parser.add_argument("--batch_size", type=int, default=100, help="Minibatch size")
parser.add_argument("--keep_prob", type=float, default = 0.5, help="Percentage (0.0-1.0) of Nodes to keep in dropout layer on each training pass")
parser.add_argument("--verbose", type=int, default=0, help="0: Almost No Output (On Script Completion), 1: Very Little Output (On Epoch completion), 2: Some Output (Every 50 steps), 3: All the Output")
parser.add_argument("--name", type=str, default=None, help="Name of the run - Provide if you want tensorboard to display multiple run (each with unique name)")
args = parser.parse_args()

n_hidden = args.n_hidden
learning_rate=args.lr
n_epochs=args.n_epochs
batch_size = args.batch_size
dropout_prob = args.keep_prob
name = args.name
verbosity = args.verbose

if verbosity > 3:
    verbosity = 3


def load_dataset():
    '''
    Loads the Tox21 Dataset (https://tox21.gov/resources/) and removes unecessary features

    Splits into training, validation, and testing data

    Return
    ------
    train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w
    '''

    # Using the [Tox21 Dataset](https://tox21.gov/resources/)
    print("Loading dataset...")
    _,(train, valid, test),_ = dc.molnet.load_tox21()

    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w

    train_y = train_y[:,0]
    valid_y = valid_y[:,0]
    test_y = test_y[:,0]
    train_w = train_w[:,0]
    valid_w = valid_w[:,0]
    test_w = test_w[:,0]
    print("Dataset split into training, validation, and testing sets")

    return train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w

def get_baseline_scores(training_data: list, validation_data: list, testing_data: list):
    '''
    Gets baseline data for the dataset using a random forest classifier implemented with sklearn.
    If weights are passed, a weighted score will be returned. A total of 3 run on each list is done.

    Parameters
    ----------
    training_data: list
        Pass a list of training X, y, w(optional)
    validation_data: list
        Pass a list of validation X, y, w(optional)
    testing_data: list
        Pass a lsit of testing data X, y, w(optional)

    Return
    ------
    scores: list[(),(),()]
        A list containing 3 tuples, each tuple contains 3 scores corresponding to 3 runs on the datasets. (i.e., (train_score, val_score, test_score)...)
    '''
    scores = []
    for i in range(3):
        model = RandomForestClassifier(class_weight="balanced",n_estimators=50)
        model.fit(training_data[0], training_data[1])

        train_y_pred = model.predict(training_data[0])
        valid_y_pred = model.predict(validation_data[0])
        test_y_pred = model.predict(testing_data[0])

        if len(training_data) == 3:
            train_score = accuracy_score(training_data[1], train_y_pred, sample_weight=training_data[2])
        else:
            train_score = accuracy_score(training_data[1], train_y_pred)

        if len(training_data) == 3:
            valid_score = accuracy_score(validation_data[1], valid_y_pred, sample_weight=validation_data[2])
        else:
            valid_score = accuracy_score(validation_data[1], valid_y_pred)

        if len(training_data) == 3:
            test_score = accuracy_score(testing_data[1], test_y_pred, sample_weight=testing_data[2])
        else:
            test_score = accuracy_score(testing_data[1], test_y_pred)

        scores.append((train_score, valid_score, test_score))

    return scores

# Defining the graph
# d = 1024 # Dimensionality of the feature vector
# n_hidden = 50
# learning_rate = .001
# n_epochs = 10
# batch_size = 100
# dropout_prob = 0.5

def eval_tox21_hyperparams(training_data: list, validation_data: list, n_hidden=50, n_layers=1, learning_rate=0.001, dropout_prob=0.5, n_epochs=45, batch_size=100, weight_positives=True, verbosity=1):

    train_X, train_y, train_w = training_data
    valid_X, valid_y, valid_w = validation_data

    d = 1024

    with tf.name_scope("placeholders"):
        x = tf.compat.v1.placeholder(tf.float32, (None, d))
        y = tf.compat.v1.placeholder(tf.float32, (None,))
        w = tf.compat.v1.placeholder(tf.float32, (None,))
        keep_prob = tf.compat.v1.placeholder(tf.float32) # Dropout placeholder

    for layer in range(n_layers):
        with tf.name_scope("layer-{}".format(layer)):
            W = tf.compat.v1.Variable(tf.compat.v1.random_normal((d, n_hidden)))
            b = tf.compat.v1.Variable(tf.compat.v1.random_normal((n_hidden,)))
            x_hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x,W) + b)
            x_hidden = tf.compat.v1.nn.dropout(x_hidden, keep_prob) # Applying dropout

    with tf.name_scope("output"):
        W = tf.compat.v1.Variable(tf.compat.v1.random_normal((n_hidden, 1)))
        b = tf.compat.v1.Variable(tf.compat.v1.random_normal((1,)))
        y_logit = tf.compat.v1.matmul(x_hidden,W) + b

        # The sigmoid gives the class probability of 1
        y_one_prob = tf.compat.v1.sigmoid(y_logit)

        # Rounding P(y=1) will give the correct prediction
        y_pred = tf.compat.v1.round(y_one_prob)

    with tf.name_scope("loss"):
        # Compute the cross-entropy term for each datapoint
        y_expand = tf.compat.v1.expand_dims(y, 1)
        entropy = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)

        # Multiply by weights
        if weight_positives:
            w_expand = tf.compat.v1.expand_dims(w,1)
            entropy = w_expand * entropy

        # Sum all contributions
        l = tf.compat.v1.reduce_sum(entropy)

    with tf.name_scope("optim"):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(l)

    with tf.name_scope("summaries"):
        tf.compat.v1.summary.scalar("loss", l)
        merged = tf.compat.v1.summary.merge_all()


    # Training the model
    hyperparam_str = f"d-{d}-hidden-{n_hidden}-lr-{learning_rate}-n_epochs-{n_epochs}-batch_size-{batch_size}-weight_pos-{str(weight_positives)}"
    filename = f'/tmp/fcnet-func-{hyperparam_str}'
    train_writer = tf.compat.v1.summary.FileWriter(filename,tf.compat.v1.get_default_graph())
    N = train_X.shape[0]

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Minibatch implementation
        step = 0
        for epoch in range(n_epochs):
            pos = 0
            while pos < N:
                batch_X = train_X[pos:pos + batch_size]
                batch_y = train_y[pos:pos + batch_size]
                batch_w = train_w[pos:pos + batch_size]
                feed_dict = {x: batch_X, y: batch_y, w: batch_w, keep_prob: dropout_prob}
                _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)

                if verbosity == 3:
                    print("epoch %d, step %d, loss %f" % (epoch, step, loss))
                elif verbosity == 2:
                    if step % 50:
                        print("epoch %d, step %d, loss %f" % (epoch, step, loss))
                train_writer.add_summary(summary, step)

                step += 1
                pos += batch_size

        valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
        weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
        if verbosity != 0:
            print("Valid Weighted Classification Accuracy: {}".format(weighted_score))
        
        return weighted_score




