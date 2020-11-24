import cv2
import numpy as np 
import dlib
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--streamType', type=str, required=True,
                help='Choose betwee -picam- and -webcam-')
args = vars(ap.parse_args())


