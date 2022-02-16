"""
Karl Estes
CSC 580 Critical Thinking 1
Created: Feburary 16th 2022
Due: Feburary 20th 2022

Asignment Prompt
----------------
For this assignment, you will write Python code to detect faces in an image.

Supply yor own image file that contains one or more faces to identify. The resulting output should be the image file with red boxes drawn around the faces

Because most human faces have roughly the same structure, the pre-trained face detection model will work well for almost any image. There's no need to trian
a new one from scratch. Use PIL wich is the Python Image Library.

File Description
----------------


Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import PIL.ImageDraw
import face_recognition
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="path to image file")

    args = parser.parse_args()

    try:
        img = np.load(args.img_path)
    except:
        print("Unable to load image from {}".args.img_path)
        exit(0)

    
