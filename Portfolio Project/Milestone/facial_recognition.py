"""
Karl Estes
CSC 580 - Module 4 Portfolio Milestone
Created: March 2nd, 2022
Due: March 13th, 2022

Asignment Prompt
----------------
# TODO - Prompt

File Description
----------------
# TODO - File Description

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import numpy as np
import face_recognition as fr
import argparse
import progressbar
import PIL
import os
import Augmentor
import glob
from colorama import init, Fore, Back, Style

init()

def print_color(s, color=Fore.WHITE, background=Back.BLACK ,brightness=Style.NORMAL, **kwargs):
    """
    Utility function wrapping the regular `print()` function 
    but with colors and brightness
    """
    print(f"{brightness}{color}{background}{s}{Style.RESET_ALL}", **kwargs)

class Faces:
    """
    The Faces class contains encodings and names for faces loaded from images (or image) from the provided filepath

    Acceptable image extensions are .jpg, .jpeg, and .png

    Parameters
    ----------
    image_path : str
        The filepath to load the face images. Can point to an image or a folder of images

    Attributes
    ----------
    image_path : str
        This is where the provided filepath is stored
    encodings : List
        A list of facial encodings created using the face_recognition library
    names : List[str]
        A list of equal length to encodings which contains corresponding names for each face.
        The name match the facial encoding at the same list index
    __augment_path : str
        An internal variable used to store the path to the a temporary augmentation folder in case image augmentation occurs
    """

    def __init__(self, image_path : str = None):

        if not os.path.isdir(image_path):
            if not (image_path.endswith(".jpg") or image_path.endswith(".jpeg") or image_path.endswith(".png")):
                print_color("WARNING: Provided filepath is not an compatable image or directory. Issues will occur when attempting to load faces for encoding.", Fore.YELLOW)
 
        self.image_path = image_path
        self.__augment_path = os.path.join(image_path, "tmp/")
        self.encodings = []
        self.names = []

    def __parse_image_name(name: str) -> str :
        """
        Cleans up the filename and returns a string to be used as the 'name' for the face encoding
        """

        sp = os.path.split(name)
        name = sp[1] if sp[1] != None else sp[0]

        no_ext = name[0:name.find(".")] # Remove extensions
        no_ext = no_ext.replace("_", " ") # Replace underscores with places

        return no_ext
    
    def __augment_images(self, num_images : int, num_samples_per_image : int):
        """
        Performs a series of augmentations on an image to hopefully increase the robustness of the facial detection

        The total number of generated images will be equal to `num_images` * `num_samples_per_image`
        
        Parameters
        ----------
        num_images : int
            The number of images that are expected to be read in
        num_samples_per_image : int
            How many augmented images to create per image read in.
        """
        pipe = Augmentor.Pipeline(source_directory=self.image_path, output_directory="tmp")

        pipe.greyscale(0.5)
        pipe.flip_left_right(0.5)
        pipe.random_brightness(0.5, 0.5, 1.5)
        pipe.random_contrast(0.5, 0.5, 1.5)

        pipe.sample(num_images * num_samples_per_image)

        return [os.path.join('tmp', name) for name in os.listdir(self.__augment_path)]

        


    def load_faces(self, path : str = None, augment : bool = False) :
        """
        Loads faces and generates encodings from the images at the provided path

        Prequisite
        ----------
        Each image to be loaded in should only have one face in them. If an image has more than one face, only the first face detected will be encoded.

        Parameters
        ----------
        path : str
            The path to load images from. WILL OVERWRITE PATH PROVIDED AT CLASS INSTANTIATION. LEAVE AS NONE TO PREVENT OVERWRITE
        augment : bool
            T/F on whether image augmentation should be performed on the loaded images to create more encodings for a face
        """

        # Check if path overwrite
        if path != None:
            self.image_path = path

        if self.image_path is None:
            print_color("ERROR: Image path is None. Unable to load faces from nothing...", Fore.RED, Back.BLACK, Style.BRIGHT)
            exit(1)

        # Get num images
        if not (os.path.isdir(self.image_path)):
            images = [self.image_path]
            num_images = 1
        else:
            images = os.listdir(self.image_path)
            num_images = len(images)

        # Augment
        if augment:
            augmented_images = self.__augment_images(len(num_images), 5)
            images = images + augmented_images
            num_images = len(images) # New length with augmented images

        # Loop until all images are loaded
        for i in progressbar.progressbar(range(num_images), redirect_stdout=True):

            image_path = os.path.join(self.image_path, images[i])

            # Ensure image is file
            if not (os.path.isfile(image_path)):
                print_color("WARNING: {} is not a file. Unable to read".format(images[i]), Fore.YELLOW)
                continue

            # Load Image File
            image = fr.load_image_file(image_path)

            # Generate Encoding
            try:
                encoding = fr.face_encodings(image)[0]
                name = self.__parse_image_name(image_path)

                self.encodings.append(encoding)
                self.names.append(name)
            except IndexError:
                print_color("WARNING: Unable to locate any faces in {}".format(images[i]), Fore.YELLOW)
            except:
                print_color("ERROR: Something went wrong when processing {}. Please check the image file. Aborting...".format(image_path), Fore.RED, Back.BLACK, Style.BRIGHT)
                exit(2)

        # Remove augmented images that might have been created
        for f in os.listdir(self.__augment_path):
            try:
                os.remove(os.path.join(self.__augment_path, f))
            except OSError as e:
                print_color(f"ERROR Deleting Augmented Image File {f} : {e}")
        

        
