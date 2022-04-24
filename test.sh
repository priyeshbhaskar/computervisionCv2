#!/bin/sh

./darknet/darknet detector train yolo-stanfordcar-data/custom_data.data yolo-stanfordcar-data/yolov3_custom_train.cfg yolo-stanfordcar-data/yolov3_custom_train_last_final.weights -dont_show
