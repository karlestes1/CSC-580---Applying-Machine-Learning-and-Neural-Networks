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

from curses import raw
from xmlrpc.client import Boolean
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import itertools
import os
import time
from tqdm import tqdm
from deepchem import deepchem as dc
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from queue import Queue
from threading import Lock, Thread
from multiprocessing import Value

# Seeds are set via assignment parameters
np.random.seed(456)
#tf.random.set_seed(456)

# Disables eager execution so TF v1 code can be run
tf.compat.v1.disable_eager_execution()

# Disable everything but error message from Tensorflow
tf.get_logger().setLevel('ERROR')



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
    for i in tqdm(range(3), desc="Random Forest - Baseline Scores"):
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

def eval_tox21_hyperparams(training_data: list, validation_data: list, n_hidden=50, n_layers=1, learning_rate=0.001, dropout_prob=0.5, n_epochs=45, batch_size=100, weight_positives=True, verbosity=1, early_stop=False):

    train_X, train_y, train_w = training_data
    valid_X, valid_y, valid_w = validation_data

    d = 1024

    # Early stopping implemented by way of accuracy on validation set
    min_delta = 0.01
    patience = 5
    history = []


    graph = tf.compat.v1.Graph()
    with graph.as_default():

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
        #hyperparam_str = f"hidden-{n_hidden}-layers-{n_layers}-lr-{learning_rate}-dropout-{dropout_prob}-n_epochs-{n_epochs}-batch_size-{batch_size}-weight_pos-{str(weight_positives)}"
        #filename = f'/tmp/tox_21/fcnet-func-{hyperparam_str}'
        #train_writer = tf.compat.v1.summary.FileWriter(filename,tf.compat.v1.get_default_graph())
        N = train_X.shape[0]

        #if verbosity == 3:
        #    print(hyperparam_str)

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
                    #train_writer.add_summary(summary, step)

                    step += 1
                    pos += batch_size

                if early_stop == True:
                    valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
                    weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)

                    if len(history) > 5:
                        #print("Hist > 5 | Val Score {} | Prev Hist Mean {} | Diff {:.3f}".format(weighted_score, np.mean(history[-patience:]), (weighted_score - np.mean(history[:-patience]))))
                        if (weighted_score - np.mean(history[-patience:])) < min_delta:
                            #print("Early Stop")
                            break

                    history.append(weighted_score)

                    



            valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
            weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
            if verbosity != 0:
                print("Valid Weighted Classification Accuracy: {}".format(weighted_score))
            
            return weighted_score

def run_model(training_data: list, validation_data: list, testing_data: list, n_hidden=50, n_layers=1, learning_rate=0.001, dropout_prob=0.5, n_epochs=45, batch_size=100, weight_positives=True, verbosity=1):
    train_X, train_y, train_w = training_data
    valid_X, valid_y, valid_w = validation_data
    test_X, test_y, test_w = testing_data

    d = 1024
    graph = tf.compat.v1.Graph()
    with graph.as_default():

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
        hyperparam_str = f"top_model--hidden-{n_hidden}-layers-{n_layers}-lr-{learning_rate}-dropout-{dropout_prob}-n_epochs-{n_epochs}-batch_size-{batch_size}-weight_pos-{str(weight_positives)}"
        filename = f'/tmp/tox_21/fcnet-func-{hyperparam_str}'
        train_writer = tf.compat.v1.summary.FileWriter(filename,tf.compat.v1.get_default_graph())
        N = train_X.shape[0]

        if verbosity == 3:
            print(hyperparam_str)

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

            # Generate Scores
            train_y_pred = sess.run(y_pred, feed_dict={x: train_X, keep_prob: 1.0})
            valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
            test_y_pred = sess.run(y_pred, feed_dict={x: test_X, keep_prob: 1.0})

            train_weighted_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
            valid_weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
            test_weighted_score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)

            if verbosity != 0:
                print("Train Weighted Classification Accuracy: {}".format(train_weighted_score))
                print("Valid Weighted Classification Accuracy: {}".format(valid_weighted_score))
                print("Test Weighted Classification Accuracy : {}".format(test_weighted_score))

            return [train_weighted_score, valid_weighted_score, test_weighted_score]

def eval_thread(thread_num, q: Queue, pbar, scores, reps, thresh, early_stop, counter, snapshot_thresh, num_models, baseline_avg):

    while not q.empty():

        n_hidden, lr, n_epochs, n_layers, batch_size, dropout, weighted_pos = q.get()
        temp_scores = []
        for i in tqdm(range(reps), desc=f"Thread {thread_num} - HPO Testing", position=thread_num, leave=False):
            temp_scores.append(eval_tox21_hyperparams([train_X, train_y, train_w], [valid_X, valid_y, valid_w], n_hidden, n_layers, lr, dropout, n_epochs, batch_size, weighted_pos, verbosity, early_stop)) # Get score from validation set

        q.task_done()

        tqdm.get_lock().acquire() # Acquire Lock
        counter.value += 1 # Update Counter
        pbar.update(1) # Update main HPO Bar
        if (baseline_avg - np.mean(temp_scores)) < thresh: # Add to score list
            scores[(n_hidden, n_layers, lr, dropout, n_epochs, batch_size, weighted_pos)] = temp_scores
        if (counter.value % snapshot_thresh) == 0: # See if need to save snapshot
            save_score_snapshot(scores, num_models, counter.value)
        tqdm.get_lock().release()

def save_score_snapshot(scores, num_models, count):

    # Check if dir exists
    if not os.path.isdir("/tmp/tox_21/logs"):
        os.makedirs("/tmp/tox_21/logs")

    with open("/tmp/tox_21/logs/top_{}_models_step_{}.txt".format(num_models, count), "w+") as file:
        tqdm.write(f"Saving snapshot of top {num_models} models at step {count} to /tmp/tox_21/logs/")
        top_model_keys = sorted(scores, key=scores.get, reverse=True)[:num_models]
        file.write("*** Step: {} | Top {} Models ***\n\n".format(count, num_models))
        for i, params in enumerate(top_model_keys):
            file.write(f"Config {i+1}: Scores = {scores[params]}\n\tNeurons per Hidden Layer: {params[0]}\n\tNumber of Hidden Layers: {params[1]}\n\tLearning Rate: {params[2]}\n\tDropout (Keep %): {params[3]}\n\tNumber of Epochs: {params[4]}\n\tBatch Size: {params[5]}\n\tWeight Positives: {str(params[6])}\n\n")

def eval_thread_writeback(thread_num, inqueue: Queue, outqueue: Queue, reps, early_stop):
    while not inqueue.empty():

        n_hidden, lr, n_epochs, n_layers, batch_size, dropout, weighted_pos = q.get()
        temp_scores = []
        for i in tqdm(range(reps), desc=f"Thread {thread_num} - HPO Testing", position=thread_num, leave=False):
            temp_scores.append(eval_tox21_hyperparams([train_X, train_y, train_w], [valid_X, valid_y, valid_w], n_hidden, n_layers, lr, dropout, n_epochs, batch_size, weighted_pos, verbosity, early_stop)) # Get score from validation set
        outqueue.put(((n_hidden, n_layers, lr, dropout, n_epochs, batch_size, weighted_pos),temp_scores))
        inqueue.task_done()

def save_score_snapshot_thread(pbar, q: Queue, scores, thresh, snapshot_thresh, num_models, baseline_avg, stop_flag):
    count = 0

    while True: # Make sure to finish queue as well
        try:
            item = q.get(timeout=5.0)
            count += 1
            pbar.update(1)

            params = item[0]
            raw_scores = item[1]

            # tqdm.write("{}".format(params))
            # tqdm.write("{}".format(raw_scores))

            if (baseline_avg - np.mean(raw_scores)) < thresh: # Add to score list if within threshold
                scores[params] = raw_scores
            
            if (count % snapshot_thresh) == 0: # See if need to save snapshot
                save_score_snapshot(scores, num_models, count)

            q.task_done()
        except:
            if stop_flag() : 
                break
            pass

    tqdm.write("Finished saving scores")

if __name__ == "__main__":
	
    # Arguments for script
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_hidden", type=int, nargs=3, default=[50,50,1], help="start stop count - Value range for number of hidden units to be tested during HPO")
    parser.add_argument("--lr", "--learning_rate", type=float, nargs=3, default=[0.001, 0.001, 1], help="low high count - Value range for learning rate to be tested during HPO")
    parser.add_argument("--n_epochs", type=int, nargs=3, default=[45,45,1], help="start stop count - Value range for the number of epochs to be tested during HPO") 
    parser.add_argument("--n_layers", type=int, nargs=3, default=[1,1,1], help="start stop count - Value range for number of hidden layers to be tested during HPO")
    parser.add_argument("--batch_size", type=int, nargs=3, default=[100,100,1], help="start stop count - Value range for batch size to be tested during HPO")
    parser.add_argument("--dropout", type=float, nargs=3, default=[0.5,0.5,1], help="low high count - Value range for dropout (keep_prob) percentage to be tested during HPO")
    parser.add_argument("-w", "--weight_positives", type=str, default=None, help="(T/True/F/False) value denoting whether to weight positive examples during training. If not provided, it is assumed both values will be tested")
    parser.add_argument("-r", "--reps", type=int, default=3, help="Number of reps to do for each hyperparameter combo. Final scores will be averaged across all reps for a specific model")
    parser.add_argument("-t", "--test_num", type=int, default=2, help="Number of final models to test against baseline after HPO. Models will be sorted by performance and the provided number will be tested")
    parser.add_argument("-e", "--early_stop", action="store_true", default=False, help="Flag denoting whether to turn on early-stopping during the HPO search training process")
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Verbosity Level 0-3")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to test HPO combinations. If <= 1, only main thread will be used. Logging of snapshot file is not enabled with multithreading")


    args = parser.parse_args()

    verbosity = args.verbose
    if verbosity > 3:
        verbosity = 3

    thresh = 0.20 # Models scores during HPO are compared with baseline. Any score with a difference below more than thresh isn't save to reduce memory for large HPO search
    snapshot_thresh = 10

    train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w = load_dataset()

    # Generate lists of HPO variables
    # Array->set->list is non-optimal but effective way to remove potential duplicates from float->int conversion
    print("Generating hyperparameter lists")
    num_hidden_list = list(set(np.random.uniform(low=args.n_hidden[0], high=args.n_hidden[1], size=args.n_hidden[2]).astype(int)))
    lr_list = np.random.uniform(low=args.lr[0], high=args.lr[1], size=int(args.lr[2]))
    num_epochs_list = list(set(np.random.uniform(low=args.n_epochs[0], high=args.n_epochs[1], size=args.n_epochs[2]).astype(int)))
    num_layers_list = list(set(np.linspace(start=args.n_layers[0], stop=args.n_layers[1], num=args.n_layers[2]).astype(int)))
    batch_size_list = list(set(np.random.uniform(low=args.batch_size[0], high=args.batch_size[1], size=args.batch_size[2]).astype(int)))
    dropout_list = np.random.uniform(low=args.dropout[0], high=args.dropout[1], size=int(args.dropout[2]))

    weight = args.weight_positives
    if weight is None:
        weighted_list = [True, False]
    else:
        weight = weight.strip() # Remove potential leading whitespace
        if weight.lower() == "t" or weight.lower() == ("true"):
            weighted_list = [True]
        elif weight.lower() == "f" or weight.lower() == ("false"):
            weighted_list = [False]
        else:
            weighted_list = [True, False]

    q = Queue(maxsize=0)
    combos = itertools.product(num_hidden_list, lr_list, num_epochs_list, num_layers_list, batch_size_list, dropout_list, weighted_list)

    # Placing all params in queue
    combos = list(combos)
    for combo in tqdm(combos,desc="Filling HPO Queue"):
        q.put(combo)

    if verbosity == 3:
        print(combos)

    # Baseline Scores
    baseline = get_baseline_scores([train_X, train_y, train_w], [valid_X, valid_y, valid_w], [test_X, test_y, test_w])
    avg_train = (baseline[0][0] + baseline[1][0] + baseline[2][0])/3.0
    avg_valid = (baseline[0][1] + baseline[1][1] + baseline[2][1])/3.0
    avg_test = (baseline[0][2] + baseline[1][2] + baseline[2][2])/3.0
    print("\nBaseline Scores:\n\tTraining Set: {:.2f}\n\tValidation Set: {:.2f}\n\t Testing Set: {:.2f}".format(avg_train, avg_valid, avg_test))

    # Basically Random HPO via Grid Search
    scores = {}
    count = 0 

    with tqdm(total=len(combos), desc="Hyperparameters", position=0) as pbar:
        while not q.empty():
            if args.num_threads <= 1:
                n_hidden, lr, n_epochs, n_layers, batch_size, dropout, weighted_pos = q.get()
                count += 1
                for reps in tqdm(range(args.reps), desc="Repetitions", position=1, leave=False):
                    pbar.update(0)
                    score = eval_tox21_hyperparams([train_X, train_y, train_w], [valid_X, valid_y, valid_w], n_hidden, n_layers, lr, dropout, n_epochs, batch_size, weighted_pos, verbosity, args.early_stop) # Get score from validation set

                    if (avg_valid - score) < thresh: # Only save to scores list if within threshold range
                        if (n_hidden, n_layers, lr, dropout, n_epochs, batch_size, weighted_pos) not in scores:
                            scores[(n_hidden, n_layers, lr, dropout, n_epochs, batch_size, weighted_pos)] = [] 
                        scores[(n_hidden, n_layers, lr, dropout, n_epochs, batch_size, weighted_pos)].append(score)
                pbar.update(1)
                q.task_done()

                if (count % snapshot_thresh) == 0:
                    save_score_snapshot(scores, args.test_num, count)
            else:
                '''
                SINGLE QUEUE
                ------------
                counter = Value('i', 0, lock=False)
                counter.value = 0
                for i in range(args.num_threads):
                    # l = Lock()
                    worker = Thread(target=eval_thread, args=(i+1, q, pbar, scores, args.reps, thresh, args.early_stop, counter, snapshot_thresh, args.test_num, avg_valid))
                    worker.start()
                q.join()
                '''
                score_queue = Queue()
                stop_flag = False
                for i in range(args.num_threads): # Start processing threads
                    worker = Thread(target=eval_thread_writeback, args=(i+1, q, score_queue, args.reps, args.early_stop))
                    worker.start()
                
                # Start save thread
                save_thread = Thread(target=save_score_snapshot_thread, args=(pbar, score_queue, scores, thresh, snapshot_thresh, args.test_num, avg_valid, lambda : stop_flag))
                save_thread.start()

                # Safely come back to main thread by waiting for everything
                q.join()
                score_queue.join()
                stop_flag = True
                save_thread.join()

        pbar.close()
        
    # Get average scores
    avg_scores = {}
    for params, param_scores in tqdm(scores.items(), desc="Averaging Scores"):
        avg_scores[params] = np.mean(np.array(param_scores))
    print("Scores averaged over {} repetitions".format(args.reps))

    if verbosity == 3:
        print(avg_scores)

    # Sort Models and get top t
    print("Getting top {} model configurations".format(args.test_num))
    top_model_keys = sorted(avg_scores, key=avg_scores.get, reverse=True)[:args.test_num]
    print(top_model_keys)

    # Compare to baseline
    print("Testing top {} models with 3 reps each")

    print("Baseline Scores:\n\tTraining Set: {:.2f}\n\tValidation Set: {:.2f}\n\t Testing Set: {:.2f}\n".format(avg_train, avg_valid, avg_test))
    with tqdm(total=len(top_model_keys), desc="Top Model Testing", position=0) as pbar:
        for params in top_model_keys:
            scores = []
            # Print hyperparemters
            tqdm.write("\nModel Config:")
            tqdm.write(f"\tNeurons per Hidden Layer: {params[0]}\n\tNumber of Hidden Layers: {params[1]}\n\tLearning Rate: {params[2]}\n\tDropout (Keep %): {params[3]}\n\tNumber of Epochs: {params[4]}\n\tBatch Size: {params[5]}\n\tWeight Positives: {str(params[6])}\n")

            for rep in tqdm(range(3), desc="Repetition", position=1, leave=False):
                s = run_model([train_X, train_y, train_w], [valid_X, valid_y, valid_w], [test_X, test_y, test_w], n_hidden=params[0], n_layers=params[1], learning_rate=params[2], dropout_prob=params[3], n_epochs=params[4], batch_size=params[5], weight_positives=params[6], verbosity=args.verbose)
                scores.append(s)

            scores = list(zip(*scores))
            avg_train = np.mean(np.array(scores[0]))
            avg_valid = np.mean(np.array(scores[1]))
            avg_test = np.mean(np.array(scores[2]))

            tqdm.write(" Scores:\n\tTraining Set: {:.2f}\n\tValidation Set: {:.2f}\n\t Testing Set: {:.2f}\n\n".format(avg_train, avg_valid, avg_test))
            pbar.update(1)
        pbar.close()

            

    

    
    
		
	
	
	


