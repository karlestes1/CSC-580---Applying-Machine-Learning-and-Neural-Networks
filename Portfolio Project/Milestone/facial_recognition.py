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
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with this extension. To trigger these keywords, they must be typed in all caps. 
"""

from contextlib import redirect_stdout
from turtle import back
import numpy as np
import face_recognition as fr
import argparse
import progressbar
import os
import sys
import math
import Augmentor
from colorama import init, Fore, Back, Style
from PIL import Image, ImageDraw

init()

def print_color(s, color=Fore.WHITE, background=None ,brightness=Style.NORMAL, **kwargs):
    """
    Utility function wrapping the regular `print()` function 
    but with colors and brightness
    """
    if background is None:
        print(f"{brightness}{color}{s}{Style.RESET_ALL}", **kwargs)
    else:
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
    encoding : List
         A list facial encoding created using the face_recognition library
    names
        A name associated with the encoding. Matched by list index 
    __augment_path : str
        An internal variable used to store the path to the a temporary augmentation folder in case image augmentation occurs
    """

    def __init__(self, image_path : str = None):

        if not os.path.isdir(image_path):
            if not (image_path.endswith(".jpg") or image_path.endswith(".jpeg") or image_path.endswith(".png")):
                print_color("WARNING: Provided filepath is not a compatable image or directory. Issues will occur when attempting to load faces for encoding.", Fore.YELLOW)
 
        self.image_path = image_path
        self.__augment_path = os.path.join(image_path, "tmp/")
        
        self.encodings = []
        self.names = []

        if not os.path.isdir(self.__augment_path):
            os.makedirs(self.__augment_path)

    def __parse_image_name(self, name: str) -> str :
        """
        Cleans up the filename and returns a string to be used as the 'name' for the face encoding
        """

        sp = os.path.split(name)
        name = sp[1] if sp[1] != None else sp[0]

        no_ext = name[0:name.find(".")] # Remove extensions
        no_ext = no_ext.replace("faces_original_", "")
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
            images.remove("tmp")
            images.remove(".DS_Store")
            num_images = len(images)

        # Augment
        if augment:
            augmented_images = self.__augment_images(num_images, 5)
            images = images + augmented_images
            num_images = len(images) # New length with augmented images

        # Loop until all images are loaded
        for i in progressbar.progressbar(range(num_images), redirect_stdout=True):

            image_path = os.path.join(self.image_path, images[i])

            # Ensure image is file
            if not (os.path.isfile(image_path)):
                print_color("WARNING: {} is not a file. Unable to read".format(images[i]), Fore.YELLOW)
                continue

            try:
                image = fr.load_image_file(image_path)
            except:
                print_color(" ERROR: Something went wrong when loading {}. Please check the image file. Aborting...".format(image_path), Fore.RED, None, Style.BRIGHT, file=sys.stderr)

            # Load Image file and generate encodings
            try:
                # Load Image File
                encoding = fr.face_encodings(image)[0]
                name = self.__parse_image_name(image_path)

                self.encodings.append(encoding)
                self.names.append(name)

                # self.faces.append({'encoding': encoding, 'name': name})
            except IndexError:
                print_color("WARNING: Unable to locate any faces in {}".format(images[i]), Fore.YELLOW)
            except:
                print_color(" ERROR: Something went wrong when processing {}. Please check the image file. Aborting...".format(image_path), Fore.RED, None, Style.BRIGHT, file=sys.stderr)
                exit(2)

        # Remove augmented images that might have been created
        for f in os.listdir(self.__augment_path):
            try:
                os.remove(os.path.join(self.__augment_path, f))
            except OSError as e:
                print_color(f"ERROR Deleting Augmented Image File {f} : {e}")

    def get_unique_names(self):
        """
        Returns a set of unique names so as to avoid duplicates from image annotation
        """
        return set(self.names)
    
    def get_face_encodings(self, name : str):
        """
        Return a list of all the face encodings that match a specific name
        """
        found = []
        
        for i, _name in enumerate(self.names):
            if name == _name:
                found.append(self.encodings[i])

        if len(found) == 0:
            print_color("WARNING: No encodings found for name {}".format(name), Fore.YELLOW, None, Style.NORMAL, file=sys.stderr)

        return found
        
class Unknown_Images:
    """
    The Unknown_Images class contains loaded copies of all images with unknown faces, locations for each face, and encodings for each face.

    Acceptable image extensions are .jpg, .jpeg, and .png

    Parameters
    ----------
    path : str
        Path to dir/file of image(s) to be loaded in

    Attributes
    ----------
    path : str
        Path to dir/file of image(s) to be loaded in
    images : list[dict(
        image : Image
            A PIL-formated image
        face_locations : list[tuple]
            A list containing all face locations (top, right, bottom, left) for the associated image. Indexes match `encodings`
        encodings : list[]
            A list containing face encodings
        filename : str
            The original name of the file without file extensions
        extension : str
            The original file extension
    )]
    """

    def __init__(self, path : str = None):
        if not os.path.isdir(path):
            if not (path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".png")):
                print_color("WARNING: Provided filepath is not a compatable image or directory. Issues will occur when attempting to load faces for encoding.", Fore.YELLOW)

        self.path = path
        self.images = []

    def load_images(self, path : str = None):
        """
        Loads faces and generates encodings for each face in the detected image

        Parameters
        ----------
        path : str
            The path to load images from. WILL OVERWRITE PATH PROVIDED AT CLASS INSTANTIATION. LEAVE AS NONE TO PREVENT OVERWRITE
        """

        # Check if path overwrite
        if path != None:
            self.path = path

        if self.path is None:
            print_color("ERROR: Image path is None. Unable to load faces from nothing...", Fore.RED, Back.BLACK, Style.BRIGHT)
            exit(1)

        # Get num images
        if not (os.path.isdir(self.path)):
            images = [self.path]
            num_images = 1
        else:
            images = os.listdir(self.path)
            num_images = len(images)

        # Loop until all images are loaded
        for i in progressbar.progressbar(range(num_images), redirect_stdout=True):

            # Load the image
            unknown_image = fr.load_image_file(os.path.join(self.path, images[i]))

            try:
                face_locations = fr.face_locations(unknown_image)
                face_encodings = fr.face_encodings(unknown_image, face_locations)
                
                self.images.append({'image': Image.fromarray(unknown_image), 'face_locations': face_locations, 'face_encodings': face_encodings, 'filename': (images[i])[0:images[i].find(".")], 'extension': (images[i])[images[i].find("."):]})

            except IndexError:
                print_color("WARNING: No faces detected in unknown image: {}. Ignooring and moving to next image...".format(images[i]), Fore.YELLOW)
                continue
            except:
                print_color(" ERROR: Something went wrong when processing {}. Please check the image file. Aborting...".format(images[i]), Fore.RED, None, Style.BRIGHT, file=sys.stderr)

                
def annotate_images(faces : Faces, images: Unknown_Images, output_dir : str) :
    """
    For each image in the Unknown_Images class, bounding boxes with names are drawn around the faces and a copy is saved to the specified `output_dir`

    Annotation code is primarily adapted from example in face_recognition library (https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py)

    Parameters
    ----------
    faces : Faces
        A collection of all the known faces. `Faces.load_faces()` should have already been run
    images : Unknown_Images
        A collection of all of the images that will be annotated. `Unknown_Images.load_images()` should have already been run
    output_dir : str
        A path to the output directory. The directory will be created if it doesn't already exist
    """

    # Check if dir exists
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            print_color("ERROR creating dir {}".format(output_dir), Fore.RED, None, Style.BRIGHT, file=sys.stderr)
            exit(4)

    # Loop for all images
    for group in progressbar.progressbar(images.images, redirect_stdout=True):

        # Create Pillow ImageDraw Draw instance to draw with
        im = group['image']
        draw = ImageDraw.Draw(im)

        # Loop through each face found in the unknown image
        for (top,right,bottom,left), unknown_encoding in zip(group['face_locations'], group['face_encodings']):
            # See if there is a face match
            matches = fr.compare_faces(faces.encodings, unknown_encoding)

            name = "Unknown"

            # Use known face with smallest distance to new face
            face_distances = fr.face_distance(faces.encodings, unknown_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces.names[best_match_index]

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

            # Draw a bounding box around the face
            draw.rectangle((left, bottom, right, top), outline=(0,0,255))
        
        # Remove the drawing library from memory as per the Pillow docs
        del draw

        # Save the image
        try:
            im.save(os.path.join(output_dir, "{}_annotated{}".format(group['filename'], group['extension'])))
        except:
            print_color(" ERROR: Could not save {}".format(os.path.join(output_dir, "{}_annotated{}".format(group['filename'], group['extension']))), Fore.RED, None, Style.BRIGHT, file=sys.stderr)

def search_for_specific_face(faces : Faces, images : Unknown_Images):

    known_encodings = None
    name = None
    found_images = []


    # Get specific face encoding
    if len(faces.encodings) > 1:
        # Print out all encodings
        names = faces.get_unique_names()

        for i, name in enumerate(names):
            print(f"{i+1}. {name}\t", end='')
            if (i%2) == 0:
                print("")
            
        # Get proper input
        user_input = -1
        while((user_input < 1) or (user_input > len(names))):
            user_input = int(input("Please choose a face to search for in the unknown images : "))

        name = list(names)[user_input - 1]
        known_encodings = faces.get_face_encodings(name) # Retrieve all matching encodings
    else:
        known_encodings = faces.encodings
        name = faces.names[0]

    # Loop through all images and search for specific face
    for group in progressbar.progressbar(images.images):
        for unknown_encoding in group['face_encodings']:
            # See if there is a match
            matches = fr.compare_faces(known_encodings, unknown_encoding)

            # See if a match exits
            if True in matches:
                found_images.append(f"{group['filename']}{group['extension']}")

    return name, found_images

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--known_faces", type=str, default="faces/", help="Path to either a folder containin multiple image files or a singluar image file. Each file should contain only one person's face.")
    parser.add_argument("--unknown_images", type=str, default="images/", help="Path to either a folder containin multiple image files or a singluar image file.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Path to output directory. When annotation mode is chosen, this is the folder where the annoted images will be saved.")
    parser.add_argument("-a", "--augment", action="store_true", default=False, help="Flag which turns on image augmentation pipeline of known faces to increase the number of facial encodings per known face")

    args = parser.parse_args()

    print("* * * * * Loading Known Faces and Images * * * * *")
    faces = Faces(args.known_faces)
    unknowns = Unknown_Images(args.unknown_images)

    # print_color("Loading faces and images...", Fore.WHITE, None, Style.BRIGHT)
    faces.load_faces(augment=args.augment)
    unknowns.load_images()

    # Operating Mode
    print("\nPlease choose a program mode:\n1. Search unknown images for a specified face\n2. Annote unknown images with detected faces")
    user_input = 0

    while user_input < 1 or user_input > 2:
        user_input = int(input("Choice : "))

    if user_input == 1:
        print("* * * * * Beginning Search * * * * *")
        name, images = search_for_specific_face(faces, unknowns)

        if(len(images) > 0):
            print(f"The face for {name} was detected in the following images: {images}")
        else:
            print(f"The face for {name} was not detected in any images")

    if user_input == 2:
        print("* * * * * Beginning Annotation * * * * *")
        annotate_images(faces, unknowns, args.output_dir)
    