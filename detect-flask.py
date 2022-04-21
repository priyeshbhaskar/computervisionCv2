import base64
import io
import os
import time
import cv2
import flask
import matplotlib
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, send_from_directory
from flask import request, send_file
from flask_cors import CORS  # comment this on deployment
from flask_cors import cross_origin
from flask_restful import Api
matplotlib.use('Agg')
import random

confthres = 0.3
nmsthres = 0.1
yolo_path = './'

car_df=pd.read_csv('./Car names and make.csv',index_col= False)
test_csv=pd.read_csv('./Annotations/Test Annotation.csv',index_col= False)
newdf=pd.merge(test_csv,car_df,on='Image class')


def get_labels(labels_path):
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    print("[INFO] loading YOLOv3 Wieghts from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr,format='JPEG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


def get_predection(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confthres:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    imageclass=""
    confidence=""
    if len(idxs) > 0:
       for i in idxs.flatten():
           (x, y) = (boxes[i][0], boxes[i][1])
           (w, h) = (boxes[i][2], boxes[i][3])
           color = [int(c) for c in COLORS[classIDs[i]]]
           cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
           text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

           imageclass=LABELS[classIDs[i]]
           confidence=confidences[i]
           print(LABELS[classIDs[i]])
           cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return image,imageclass,confidence

labelsPath="yolo-stanfordcar-data/classes.names"
cfgpath="yolo-stanfordcar-data/yolov3_custom_test.cfg"
wpath="yolo-stanfordcar-data/yolov3_custom_train_last.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)
Colors=get_colors(Lables)
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

app = Flask(__name__, static_url_path='', static_folder='black-dashboard-react/build')
CORS(app)
api = Api(app)

@app.route("/", defaults={'path':''})
def serve(path):
    return send_from_directory(app.static_folder,'index.html')


@app.route('/api/getClassName', methods=['GET'])
@cross_origin()
def getCarClass():
    args = request.args
    imageid = args.get('imageid')
    row= newdf[newdf['Image Name'] ==imageid]
    className=""
    if row['Class Name'].values:
        className = row['Class Name'].values[0]
    else:
        className = "N/A"
    return flask.jsonify({'className': className})


def loadImage(row, axis):
    image_path = f"/Users/pribhask/priyeshproj/capstone/cars_test/{row['Image Name']}"
    img = mpimg.imread(image_path)
    axis.imshow(img)
    lbl = row['Class Name']
    x_car0 = row['x0']
    y_car0 = row['y0']
    x_car1 = row['x1']
    y_car1 = row['y1']
    axis.add_patch(patches.Rectangle((x_car0, y_car0), x_car1 - x_car0, y_car1 - y_car0, linewidth=2, edgecolor='blue',
                                     facecolor='none'))

    axis.set_title(lbl)


def loadImages(df):
    cols = 5
    rows = 4
    idx = 0
    f, axarr = plt.subplots(rows, cols, figsize=(20, 15))
    randomsamplelist = random.sample(range(0, newdf.shape[0]), 20)
    for r in range(rows):
        for c in range(cols):
            axis = axarr[r, c]
            loadImage(df.iloc[randomsamplelist[idx]], axis)
            idx += 1
    plt.savefig('data.jpeg')







@app.route('/api/getImageBoundingBox', methods=['GET'])
@cross_origin()
def carImagesWithBoundingBox():
    loadImages(newdf)
    return send_file('data.jpeg', mimetype='image/jpeg')




# route http posts to this method
@app.route('/api/test', methods=['POST'])
@cross_origin()
def main():

    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    res,imgclass,confidence=get_predection(image,nets,Lables,Colors)

    image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    np_img=Image.fromarray(image)
    img_encoded=image_to_byte_array(np_img)
    encoded_img = "data:image/jpeg;base64,"+base64.encodebytes(img_encoded).decode('ascii')
    resp={
        "image":encoded_img,
        "imgclass":imgclass,
        "confidence":confidence
    }
    response = flask.jsonify({'response': resp})


    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8090)