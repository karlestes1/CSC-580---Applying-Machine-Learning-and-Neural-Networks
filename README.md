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

### (WIP) Critical Thinking 3: [Predicting Fuel Efficiency Using TensorFlow](CT%203)
- In a nutshell, this is a *regression* problem where a **neural network** will be created with the ```tf.keras``` API and will utilize the [**Auto MPG**](https://archive.ics.uci.edu/ml/datasets/auto+mpg) dataset. The trained model will be used to predict the feul efficiency of late 1970s and early 1980s automobiles
___
## Portfolio Project (WIP)
**Information will be added as progress is made**