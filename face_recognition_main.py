from PIL import Image, ImageTk
from imutils import paths
from matplotlib import pyplot as plt
from tkinter import Tk, Label, Menu, Toplevel, filedialog
import copy
import cv2
import face_recognition
import numpy as np
import os
import pickle

def noise_reduce(img):
    new_image = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
    return new_image

def detect_edge(img):
    new_image = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    new_image = cv2.Canny(new_image,80,140)
    return new_image

def recognition_face(img):
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.4)
        name = ""
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    face_detect_image = copy.deepcopy(img)
    face_recognize_image = copy.deepcopy(img)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(face_detect_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(face_recognize_image, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(face_recognize_image, name, (left, bottom + 50), font, 2.0, (0, 0, 255), 2)
    return face_detect_image, face_recognize_image



known_face_encodings = pickle.load(open('face_encodings.txt', 'rb'))


known_face_names = [
    "N.Vu", "N.Vu", "N.Vu", "N.Vu", "N.Vu",
    "D.Vu", "D.Vu", "D.Vu", "D.Vu", "D.Vu",
    "Bao", "Bao", "Bao", "Bao", "Bao",
    "Ky", "Ky", "Ky", "Ky", "Ky",
    "Thai", "Thai", "Thai", "Thai", "Thai"
]


image_name = 'intro.jpg'
image = None
image_rgb = None
image_gray = None

image_less_noise = None
image_detect_edge = None
image_detect_face = None
image_recognize_face = None


def reload_processing_data():
    global image, image_rgb, image_gray, \
            image_less_noise, image_detect_edge, \
            image_detect_face, image_recognize_face

    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_less_noise = noise_reduce(image_rgb)
    image_detect_edge = detect_edge(image_less_noise)
    image_detect_face, image_recognize_face = recognition_face(image_less_noise)


reload_processing_data()


root = Tk()


im = Image.fromarray(image_rgb)
width, height = im.size
imgtk = ImageTk.PhotoImage(image=im.resize([int(600 * width / height), 600]))

panel = Label(root, image=imgtk)
panel.pack(side="bottom", fill="both", expand="yes")


def callback():
    global image_name
    image_name = filedialog.askopenfilename()
    reload_processing_data()

    im = Image.fromarray(image_rgb)
    width, height = im.size
    imgtk = ImageTk.PhotoImage(image=im.resize([int(600 * width / height), 600]))

    panel.configure(image=imgtk)
    panel.image = imgtk

def original():
    im = Image.fromarray(image_rgb)
    width, height = im.size
    imgtk = ImageTk.PhotoImage(image=im.resize([int(600 * width / height), 600]))

    panel.configure(image=imgtk)
    panel.image = imgtk

def noiseReduce():
    im = Image.fromarray(image_less_noise)
    width, height = im.size
    imgtk = ImageTk.PhotoImage(image=im.resize([int(600 * width / height), 600]))

    panel.configure(image=imgtk)
    panel.image = imgtk


def edgeDetect():
    im = Image.fromarray(image_detect_edge)
    width, height = im.size
    imgtk = ImageTk.PhotoImage(image=im.resize([int(600 * width / height), 600]))

    panel.configure(image=imgtk)
    panel.image = imgtk

def faceDetect():
    im = Image.fromarray(image_detect_face)
    width, height = im.size
    imgtk = ImageTk.PhotoImage(image=im.resize([int(600 * width / height), 600]))

    panel.configure(image=imgtk)
    panel.image = imgtk


def faceRecognize():
    im = Image.fromarray(image_recognize_face)
    width, height = im.size
    imgtk = ImageTk.PhotoImage(image=im.resize([int(600 * width / height), 600]))

    panel.configure(image=imgtk)
    panel.image = imgtk


menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open Image", command=callback)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

actions = Menu(menubar, tearoff=0)
actions.add_command(label="Original", command=original)
actions.add_command(label="Noise Reduce", command=noiseReduce)
actions.add_command(label="Edge Detect", command=edgeDetect)
actions.add_command(label="Face Detect", command=faceDetect)
actions.add_command(label="Face Recognize", command=faceRecognize)
menubar.add_cascade(label="Actions", menu=actions)

root.config(menu=menubar)
root.mainloop()
