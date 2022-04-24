import tkinter as tk
import os
import pandas as pd
import PIL.Image
import subprocess
from subprocess import Popen, PIPE
import time


from tkinter import *

from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

fileNameRegression = ""
fileNameClassification = ""

root = Tk()
root.title("Object Detection Yolo Training GUI")

fileName = Label(root, text="Step 1: Folder Name:")
fileName.grid(row=0, column=0)
e = Entry(root, width=12)
e.grid(row=0, column=1)
entryText = tk.StringVar()
found = Entry(root, textvariable=entryText)
found.grid(row=0, column=4)
regressiondf = None
full_path_to_csv = './Annotations/Test Annotation.csv'
full_path_to_images = './'



def preparetrainAndValidationFile():
    p = []
    for current_dir, dirs, files in os.walk('.'):
        # Going through all files
        for f in files:
            # Checking if filename ends with '.jpg'
            if f.endswith('.jpg'):
                # Preparing path to save into train.txt file
                # Pay attention!
                # If you're using Windows, it might need to change
                # this: + '/' +
                # to this: + '\' +
                # or to this: + '\\' +
                path_to_save_into_txt_files = full_path_to_images + '/' + f

                # Appending the line into the list
                # We use here '\n' to move to the next line
                # when writing lines into txt files
                p.append(path_to_save_into_txt_files + '\n')
    p_test = p[:int(len(p) * 0.15)]
    p = p[int(len(p) * 0.15):]
    with open('train.txt', 'w') as train_txt:
        # Going through all elements of the list
        for e in p:
            # Writing current path at the end of the file
            train_txt.write(e)
    with open('test.txt', 'w') as test_txt:
        # Going through all elements of the list
        for e in p_test:
            # Writing current path at the end of the file
            test_txt.write(e)
    pass


#Train on simple darknet c library whic
# Cpu , opencv

def preprocessyolov3Data(folder):
    labelscsv = "/Users/pribhask/priyeshproj/capstone/Car names and make.csv"
    labelsdf = pd.read_csv(labelscsv)
    labels = list(labelsdf['Class Name'])
    annotations = pd.read_csv(full_path_to_csv,usecols=['Image Name','x0','y0','x1','y1','Image class'])
    sub_ann = annotations.copy()
    sub_ann['center x'] = ''
    sub_ann['center y'] = ''

    sub_ann['center x'] = (sub_ann['x0'] + sub_ann['x1']) / 2
    sub_ann['center y'] = (sub_ann['y0'] + sub_ann['y1']) / 2
    sub_ann['width'] = sub_ann['x1'] - sub_ann['x0']
    sub_ann['height'] = sub_ann['y1'] - sub_ann['y0']
    r = sub_ann.loc[:, ['Image Name','Image class','center x','center y','width','height']].copy()
    full_path_to_image=full_path_to_images+folder
    os.chdir(full_path_to_image)
    for current_dir, dirs, files in os.walk('.'):
        # Going through all files
        for f in files:
            # Checking if filename ends with '.jpg'
            if f.endswith('.jpg'):
                # Slicing only name of the file without extension
                image_name = f
                # Getting Pandas dataFrame that has only needed rows
                # By using 'loc' method we locate needed rows
                # that satisfies condition 'classes['ImageID'] == image_name'
                # that is 'find from the 1st column element
                # that is equal to image_name'
                sub_r = r.loc[r['Image Name'] == image_name]

                # Getting resulted Pandas dataFrame that has only needed columns
                # By using 'loc' method we locate here all rows
                # but only specified columns
                # By using copy() we create separate dataFrame
                # not just a reference to the previous one
                # and, in this way, initial dataFrame will not be changed
                img = PIL.Image.open(image_name)

                resulted_frame = sub_r.loc[:, ['Image class',
                                               'center x',
                                               'center y',
                                               'width',
                                               'height']].copy()
                resulted_frame['Image class'] = resulted_frame['Image class'] - 1
                resulted_frame['center x'] = resulted_frame['center x'] / img.width
                resulted_frame['center y'] = resulted_frame['center y'] / img.height
                resulted_frame['width'] = resulted_frame['width'] / img.width
                resulted_frame['height'] = resulted_frame['height'] / img.height
                path_to_save = full_path_to_image + '/' + f[:5] + '.txt'
                resulted_frame.to_csv(path_to_save, header=False, index=False, sep=' ')

    preparetrainAndValidationFile()
    pass

def loadData():
    folder=full_path_to_images+e.get()
    if os.path.isdir(folder):
        preprocessyolov3Data(e.get())
        entryText.set("Input Preprocessing Complete")

    else:
        entryText.set("Folder Not Found")

def starttraining():
    path_to_output_file = './yolotrainingoutput.txt'
    myoutput = open(path_to_output_file, 'w+')
    trainingstatus.set(f"Training Started . Please check file {path_to_output_file}")
    time.sleep(10)
    session = subprocess.Popen(['./test.sh'], stdout=myoutput, stderr=myoutput, universal_newlines=True)
    trainingstatus.set(f"Training Started . Please check file {path_to_output_file}")
    output, errors = session.communicate()



button = Button(root, text="Preprocess Yolov3 data", command=loadData)
button.grid(row=0, column=2)

trainlabel = Label(root, text="Step2: Object Detection Training")
trainlabel.grid(row=1, column=0)

tbbutton = Button(root, text="Train", command=starttraining)
tbbutton.grid(row=1, column=1)


trainingstatus = tk.StringVar()
trainingstatusLabel = Entry(root, textvariable=trainingstatus)
trainingstatusLabel.grid(row=1, column=4)

root.mainloop()