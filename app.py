from flask import Flask, render_template, Response
from flask import request, json, jsonify
import base64
import os
import time

import sys
sys.path.append('../')
import argparse
import os
import glob
import sys
import timeit
import tensorflow as tf
import numpy as np
from scipy import misc
from model_mobilev2_shelf_2 import Segment2
from tools import decode_labels
from inference_pb import get_arguments, load_img, preprocess, check_input
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ALLOWED_EXTENSIONS = ['jpg', 'png', 'bmp', 'jpeg']
sun_class = 14
num_class = 14
IMG_SHAPE = (480, 640)
IMG_PATH_OUTPUT = 'static/res.jpg'
IMG_PATH_INPUT = 'static/upload.jpg'
PB_PATH = '../pb/SegmentModel.pb'

class tfServer(object):
    def __init__(self):
        self.shape = IMG_SHAPE
        self.num_classes = sun_class
        self.show_classes = num_class
        output_graph_path = PB_PATH

        # with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        self.sess =  tf.Session()
        tf.initialize_all_variables().run(session=self.sess)
        self.img_ph = self.sess.graph.get_tensor_by_name("Placeholder:0")
        raw_output = self.sess.graph.get_tensor_by_name("final_out:0")
        raw_output_up = tf.image.resize_bilinear(raw_output, size=self.shape, align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, self.shape[0], self.shape[1])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        self.pred = decode_labels(raw_output_up, self.shape, self.show_classes, self.num_classes)

    def func(self, img_name):
        img_in, filename = load_img(img_name) # 480 640
        preds = self.sess.run(self.pred, feed_dict={self.img_ph: [img_in]})
        misc.imsave(IMG_PATH_OUTPUT, preds[0])
        img_base64 = self.numpy2base64(preds[0])
        # imgd = cv2.imread(IMG_PATH_OUTPUT)
        # cv2.imshow("imgd", imgd)
        # cv2.waitKey(0)
        return img_base64

    def base642numpy(self, img_base64):
        img_data = base64.b64decode(img_base64)
        img_np = np.fromstring(img_data, np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return img_np

    def numpy2base64(self, img_np):
        img_base64 = cv2.imencode('.jpg', img_np)[1].tostring() # <class 'bytes'>
        img_base64 = base64.b64encode(img_base64) # <class 'bytes'>
        img_base64 = img_base64.decode() # str
        return img_base64

    def __del__(self):
        self.sess.close()


app = Flask(__name__) 
ser = tfServer()

@app.route('/') 
def index_upload():
    return render_template('upload.html')

# save the image as a picture
@app.route('/img_upload', methods=['POST'])
def img_upload():
    req_file = request.files['imgFile']  # get the image
    print(type(req_file),req_file) # <class 'werkzeug.datastructures.FileStorage'> <FileStorage: 'blob' ('image/jpeg')>
    name_sub = req_file.filename.split('.')[1].lower()
    if name_sub not in ALLOWED_EXTENSIONS:
        res_str = "request image error: format not support"
        return jsonify({'status':res_str })
    img_name = 'static/upload.{}'.format(name_sub)
    req_file.save(img_name)

    start_time = time.time()
    img_bs64 = ser.func(img_name)
    print("api cost: ", time.time()-start_time)
    
    # img_f = open(res_name, 'rb')
    # bs64data = base64.b64encode(img_f.read()).decode()
    # img_f.close()
    return jsonify({'viz': img_bs64, 'status':str(time.time()-start_time)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001 , threaded=True)