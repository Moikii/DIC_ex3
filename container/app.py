# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
from PIL import ImageDraw
import os
import detect
import tflite_runtime.interpreter as tflite
import platform
import datetime
import cv2
import time
import numpy as np
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify
import random
import re


app = Flask(__name__)



def detection_loop(filename_image, path, output):
   pass


#initializing the flask app
app = Flask(__name__)

#routing http posts to this method
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
  #img = request.files["image"].read()
  #image = Image.open(io.BytesIO(img))
  #data_input = request.args['input']
  data_input = request.values.get('input')
  output = request.values.get('output')
  #output = request.form.get('output')

  path = data_input
  filename_image = {}
  
  input_format = ["jpg", "png", "jpeg"]
  if data_input.find(".") != -1:
      print(data_input + " is a file")
      split_data_input = data_input.split(".", 1)
      if data_input.endswith(tuple(input_format)):
          print("INPUT FORMAT: %s IS VALID" % split_data_input[1])
          path_splitted = []
          path_splitted = re.split('/', data_input)
          filename = path_splitted[len(path_splitted)-1]
          filename_image[filename] = Image.open(data_input)
          path = os.path.dirname(data_input)+"/"
  else:
      print(data_input + " is a path with the following files: ")
      for filename in os.listdir(data_input):
          image_path = data_input + filename
          filename_image[filename] = Image.open(image_path)
          print("  " + filename)
  
  detection_loop(filename_image, path, output)
  
  status_code = Response(status = 200)
  return status_code
# image=cv2.imread(args.input)
# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
