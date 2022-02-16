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
This script takes a single image file, detects faces in the image using the face_recognition python package, and draws a red bounding box around each face. 

The image should be shown on the screen (Note: Some issues may occur with macOS and some Ubuntu installations according to some forum threads I read). Regardless
of the image appearing on screen, a copy of the image is saved locally as 'image_out.jpg'

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
    parser.add_argument("--img_path", default="image.jpg", type=str, help="path to image file")

    args = parser.parse_args()

    # Load the image file into jumpy array
    try:
        img = np.array(PIL.Image.open(args.img_path))
    except:
        print("Unable to load image from {}".args.img_path)
        exit(0)

    print("WARNING: There are known issues with displaying images with PIL on macOS and some versions of Ubuntu. If the image does not display automatically, do not fret! A copy of the image with the bounding boxes applied will be saved.")
    # Find all faces in the image
    face_locations = face_recognition.face_locations(img)
    print("Found {} face(s) in the image".format(len(face_locations)))

    pil_image = PIL.Image.fromarray(img)
    draw = PIL.ImageDraw.Draw(pil_image)

    for face in face_locations:
        print("Face was found at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(face[0],face[1],face[2],face[3]))
        draw.rectangle([face[1], face[0], face[3], face[2]], outline='red')

    pil_image.show()
    pil_image.save("image_out.jpg")
    print("Image with bounding boxes has been saved as image_out.jpg")
    

