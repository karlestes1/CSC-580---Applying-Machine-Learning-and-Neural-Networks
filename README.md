# CSC 580 - Applying Machine Learning and Neural Networks (Captsone)
**Disclaimer:** These projects were built as a requirement for CSC 580: Applying Machine Learning and Neural Networks at Colorado State University Global under the instruction of Dr. Brian Holbert. Unless otherwise noted, all programs were created to adhere to explicit guidelines as outlined in the assignment requirements I was given. Descriptions of each [programming assignment](#programming-assignments) and the [portfolio project](#portfolio-project) can be found below.

*****This class is in progress and the respository will be updated as assignments are completed.*****
___

### Languages and Tools
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/python.svg" />](https://www.python.org)
[<img align="left" height="32" width="32" src="https://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/anaconda_navigator/static/images/anaconda-icon-512x512.png" />](https://www.anaconda.com/)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/visual-studio-code.svg" />](https://code.visualstudio.com)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/git-icon.svg" />](https://git-scm.com)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/gitkraken.svg" />](https://www.gitkraken.com)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/tensorflow.svg" />](https://www.tensorflow.org)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/jupyter.svg" />](https://jupyter.org)
<br />

### Textbooks
The required textbook for this class was [**TensorFlow for Deep Learning: From Linear Regression to Reinforcement Learning**](https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/) by **Bharath Ramsundar & Reza Bosagh Zadeh**

The optional textbook for this class was [**Deep Learning Pipeline: Building a Deep Learning Model with TensorFlow**](https://www.oreilly.com/library/view/deep-learning-pipeline/9781484253496/) by **Hisham El-Amir & Mahmoud Hamdy**
### VS Code Comment Anchors Extension
I am using the [Comment Anchors extension](https://marketplace.visualstudio.com/items?itemName=ExodiusStudios.comment-anchors) for Visual Studio Code which places anchors within comments to allow for easy navigation and the ability to track TODO's, code reviews, etc. You may find the following tags intersperesed throughout the code in this repository: ANCHOR, TODO, FIXME, STUB, NOTE, REVIEW, SECTION, LINK, CELL, FUNCTION, CLASS

For anyone using this extension, please note that CELL, FUNCTION, and CLASS are tags I defined myself. 

### Jupyter Notebooks
For many of the assignments, I used a jupyter notebook to breakdown the program and test various components of it. In cases where this was the case, I left the Jupyter Notebook uploaded. The python file is the final code for each assignment, though. 
<br />

___
## Programming Assignments
### Critical Thinking 1: [Basic Facial Recognition Program](CT%201)
- A python script to perform basic facial recognition in an image and draw a red bounding box around detected faces
- Utilizes the [facial_recognition](https://pypi.org/project/face-recognition/) and [PIL](https://en.wikipedia.org/wiki/Python_Imaging_Library) libraries

### Critical Thinking 2: [Predicting Future Sales](CT%202)
- In a nutshell, *sales_data_test.csv* and *sales_data_test.csv* contain data that will be used to train a neural network to predict how much money can be expected form the future sale of new video games. The .csv files were retrieved from one of [Toni Esteves repos](https://github.com/toniesteves/adam-geitgey-building-deep-learning-keras/tree/master/03). 
- This script produces two different models for predicting future sales of a video game. Both models are constructed with Keras and are Sequential models. One model 
was constructed via the assignment parameters and was constructed of a few Dense hidden layers with a single output node. The other model was 'optimized' using the 
keras_tuner library. 
    - To run the script with hyperparameter tuning, add ```--hpo``` when running the program
- All the data preprocessing and outputs are structured as per the assignment parameters. The inclusion of hyperparemter tuning via the keras_tuner library
was a deviation from the assignment instructions. The core assingment was still completed, however, and this gave me a chance to exlplore hyperparemter tuning.

### Critical Thinking 3: [Predicting Fuel Efficiency Using TensorFlow](CT%203)
- In a nutshell, this is a *regression* problem where a **neural network** will be created with the ```tf.keras``` API and will utilize the [**Auto MPG**](https://archive.ics.uci.edu/ml/datasets/auto+mpg) dataset. The trained model will be used to predict the feul efficiency of late 1970s and early 1980s automobiles.

### Critical Thinking 4: [Toxicology Testing](CT%204)
- In a nutshell, this is a *classification* problem where a **neural network** was tasked with classifying compounds in the [**Tox21**](https://tox21.gov/resources/) dataset as poisonous or non-poisonous.
- The assignment required that a dropout layer be added to the network and accuracy calculations implemented. 
- Rudimentary hyperparameter tuning was implemented to try and tune the number of hidden units, learning rate, number of epohcs, and dropout probability

### Critical Thinking 5: [Improving the Accuracy of a Neural Network](CT%205)
- This assignment was an extension of **Critical Thinking 4** with the goal of implementing robust hyperparameter tuning. While the assignment only required a series of for loops to test various hyperparemter combinations, I opted instead to implement a multithreaded tuning approach. 
- The script can be run in single threaded mode or multi-threaded mode
    - In multi-threaded mode, all the possible hyperparemter combinations are placed in a thread-safe Queue, and each thread pulls from the queue and test the HP config
- Early stopping can be turned on to end training early on configurations that plateau or decrease in accuracy
- Results are saved periodically during the training process with the top configurations
- Program arguments are as follows:
```
usage: tox21_hpo.py [-h] [--n_hidden N_HIDDEN N_HIDDEN N_HIDDEN] [--lr LR LR LR] [--n_epochs N_EPOCHS N_EPOCHS N_EPOCHS] 
[--n_layers N_LAYERS N_LAYERS N_LAYERS] [--batch_size BATCH_SIZE BATCH_SIZE BATCH_SIZE] [--dropout DROPOUT DROPOUT DROPOUT] 
[-w WEIGHT_POSITIVES] [-r REPS] [-t TEST_NUM] [-e] [-v] [--num_threads NUM_THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --n_hidden N_HIDDEN N_HIDDEN N_HIDDEN
                        start stop count - Value range for number of hidden units to be tested during HPO
  --lr LR LR LR, --learning_rate LR LR LR
                        low high count - Value range for learning rate to be tested during HPO
  --n_epochs N_EPOCHS N_EPOCHS N_EPOCHS
                        start stop count - Value range for the number of epochs to be tested during HPO
  --n_layers N_LAYERS N_LAYERS N_LAYERS
                        start stop count - Value range for number of hidden layers to be tested during HPO
  --batch_size BATCH_SIZE BATCH_SIZE BATCH_SIZE
                        start stop count - Value range for batch size to be tested during HPO
  --dropout DROPOUT DROPOUT DROPOUT
                        low high count - Value range for dropout (keep_prob) percentage to be tested during HPO
  -w WEIGHT_POSITIVES, --weight_positives WEIGHT_POSITIVES
                        (T/True/F/False) value denoting whether to weight positive examples during training. If not provided, it is assumed both values will be tested
  -r REPS, --reps REPS  Number of reps to do for each hyperparameter combo. Final scores will be averaged across all reps for a specific model
  -t TEST_NUM, --test_num TEST_NUM
                        Number of final models to test against baseline after HPO. Models will be sorted by performance and the provided number will be tested
  -e, --early_stop      Flag denoting whether to turn on early-stopping during the HPO search training process
  -v, --verbose         Verbosity Level 0-3
  --num_threads NUM_THREADS
                        Number of threads to test HPO combinations. If <= 1, only main thread will be used. Logging of snapshot file is not enabled with multithreading
```

### Critical Thinking 6: [CIFAR10 with CNNs](CT%206)
- A python script which utilizes the [Keras Tuner](https://keras.io/keras_tuner/) to optimize four different types of CNN on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset
    - CNN with Pooling Layers 
    - CNN without Pooling Layers
    - CNN with Dilation and Pooling Layers
    - CNN with Dilation and without Pooling Layers
- Accuracy was the only metric used to train the networks

___
## Portfolio Project
### [Milestone: Implementing Facial Recognition](Portfolio%20Project/Milestone/)
- Assignment built on the work from [Critical Thinking 1](CT%201) and required the creation of a program that could determine if an individual face was present in a group of faces. 
- Program can operated in two modes:
    - Search all unkown images for a specific face
    - Annotate all faces in the unkown images with a name or unkown label
- Face identification and comparisons were done using the [face-recognition](https://pypi.org/project/face-recognition/) library
- Image augmentation can be toggled on to increase the number of `known_faces` to compare against
    - Augmentation was implemented via the [Augmentor](https://pypi.org/project/Augmentor/) library 
- Program arguments are as follows:
```
usage: facial_recognition.py [-h] [--known_faces KNOWN_FACES] [--unknown_images UNKNOWN_IMAGES] [--output_dir OUTPUT_DIR] [-a]

optional arguments:
  -h, --help            show this help message and exit
  --known_faces KNOWN_FACES
                        Path to either a folder containin multiple image files or a singluar image file. Each file should contain only one person's face.
  --unknown_images UNKNOWN_IMAGES
                        Path to either a folder containin multiple image files or a singluar image file.
  --output_dir OUTPUT_DIR
                        Path to output directory. When annotation mode is chosen, this is the folder where the annoted images will be saved.
  -a, --augment         Flag which turns on image augmentation pipeline of known faces to increase the number of facial encodings per known face
```

### [Final: Working with a Generative Adversarial Network](Portfolio%20Project/Final/)
- The final project for this course was divided into two parts
    - **Part One: [Research Write Up](Portfolio%20Project/Final/GAN_Paper.pdf)**
        - A in depth look at four pertinent industry use cases for GANs and the benefit of using it for each
        - A n overview of generative adversarial networks is provided at the beginning of the paper
        - The four uses cases that are covered are:
            - Medical Image Synthesis
            - Controlled Image Generation
            - Super Resolution Image Scaling
            - Synthetic Network Flow Generation
    - **Part Two: [Programming Implementation](Portfolio%20Project/Final/main.py)**
        - The assignment required implementing a GAN to learn to generate images from a random class in the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. 
        - The file [`program_analysis.pdf`](Portfolio%20Project/Final/Program_Analysis.pdf) discusses some of the issues that were encountered along the way and changes that had to be made to provided code to get an output
        - The Generator model and samples of the Generator output are saved every 10 epochs
        - Example of images created by the GAN after being trained on CIFAR10 horse images:


            ![Images of horses generated from GAN](https://github.com/karlestes1/CSC-580---Applying-Machine-Learning-and-Neural-Networks/blob/main/Portfolio%20Project/Final/plots/final.jpg?raw=true)
