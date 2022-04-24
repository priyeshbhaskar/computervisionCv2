#!/bin/sh

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

${SCRIPTPATH}/darknet/darknet detector train ${SCRIPTPATH}/yolo-stanfordcar-data/custom_data.data ${SCRIPTPATH}/yolo-stanfordcar-data/yolov3_custom_train.cfg ${SCRIPTPATH}/yolo-stanfordcar-data/yolov3_custom_train_last_final.weights -dont_show
