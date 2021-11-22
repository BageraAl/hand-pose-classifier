from utils import detector_utils as detector_utils
from utils import pose_classification_utils as classifier
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import os; 
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import numpy as np
from flask import Flask
from flask import request


app = Flask(__name__)
poses = []
input_q = Queue(maxsize=5)
inferences_q = Queue(maxsize=5)

score_thresh = 0.18


def worker(input_q,inferences_q, cap_params):
    print(">> loading detection  model")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)

    print(">> loading classification model")
    try:
        model, classification_graph, session = classifier.load_KerasGraph("cnn/models/hand_poses_wGarbage_10.h5")
    except Exception as e:
        print(e)

    while True:
        frame = input_q.get()
        if (frame is not None):
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # get region of interest
            res = detector_utils.get_box_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                                               scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)

            # classify hand pose
            if res is not None:
                class_res = classifier.classify(model, classification_graph, session, res)
                inferences_q.put(class_res)
            else:
                inferences_q.put(np.array([]))

    sess.close()


@app.route("/",methods=['POST'])
def upload_file():
    print(request.files)
    if 'file' not in request.files:
        return 'No File Found'
    file = 'file'
    user_file = request.files[file]
    image_bytes = user_file.read()
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    if user_file:
        input_q.put(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        inferences = None
        try:
            while inferences is None:
                inferences = inferences_q.get()
        except Exception as e:
            pass
        return poses[np.argmax(inferences)] if inferences.size > 0  else "None"

    else:
        return 'File not found'




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=1,
        help='Number of workers.')
    args = parser.parse_args()




    # Count number of files to increment new example directory
    _file = open("poses.txt", "r")
    lines = _file.readlines()
    for line in lines:
        line = line.strip()
        if (line != ""):
            print(line)
            poses.append(line)

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'] = args.width
    cap_params['im_height'] = args.height
    print(cap_params['im_width'], cap_params['im_height'])
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    pool = Pool(args.num_workers, worker,
                (input_q, inferences_q, cap_params))

    app.run()



