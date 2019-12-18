from flask import Flask, render_template, Response
from flask import request, json, jsonify
import base64
import os
import time

ALLOWED_EXTENSIONS = ['jpg', 'png', 'bmp', 'jpeg']

class Server(object):
    def __init__(self):
        pass

    def func(self, img_name): 
        res_name = img_name
        return res_name

    def __del__(self):
        pass


app = Flask(__name__) 
ser = Server()

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
    res_name = ser.func(img_name)
    print("api cost: ", time.time()-start_time)
    
    img_f = open(res_name, 'rb')
    bs64data = base64.b64encode(img_f.read()).decode()
    img_f.close()
    return jsonify({'viz': bs64data, 'status':str(time.time()-start_time)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001 , threaded=True)