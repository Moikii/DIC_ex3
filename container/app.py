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

from urllib import response
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import os

from tqdm import tqdm
from flask import Flask, request, Response, jsonify, abort
import re
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from time import time
import json
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


#module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

detector = hub.load(module_handle).signatures['default']



def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                            int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


def detection_loop(filename_image, as_json = False):
    all_result = {}
    inference_time = 0
    for filename, image in tqdm(filename_image.items()): 
        converted_img  = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
        print(filename)
        start_time = time()
        result = detector(converted_img)
        end_time = time()

        inference_time = inference_time + end_time - start_time

        result = {key:value.numpy() for key,value in result.items()}

        print("Found %d objects." % len(result["detection_scores"]))

        if as_json: 
            result_dict = {
                k: result[k] for k in ['detection_boxes', 'detection_scores']
            }
            result_classes = [class_name.decode("ascii") for class_name in result['detection_class_entities']]
            result_dict['detection_class_entities'] = result_classes
            all_result[filename] = result_dict
        
        else: 
            image_with_boxes = draw_boxes(
                np.array(image), result["detection_boxes"],
                result["detection_class_entities"], result["detection_scores"])

            all_result[filename] = image_with_boxes
        
    return all_result, inference_time


#initializing the flask app
app = Flask(__name__)

@app.route('/api/detect', methods=['POST', 'GET'])
def main():
  data_input = request.values.get('input')

  path = data_input
  filename_image = {}
  
  input_format = ["jpg", "png", "jpeg"]
  if data_input.find(".") != -1:
      print(data_input + " is a file")
      split_data_input = data_input.split(".", 1)
      if data_input.endswith(tuple(input_format)):
          print("INPUT FORMAT: %s IS VALID" % split_data_input[1])
          path_splitted = re.split('/', data_input)
          filename = path_splitted[-1]
          filename_image[filename] = Image.open(data_input).convert('RGB')
  else:
      print(data_input + " is a path with the following files: ")
      for filename in os.listdir(data_input):
          image_path = data_input + filename
          filename_image[filename] = Image.open(image_path).convert('RGB')
          print("  " + filename)
  
  result, inference_time = detection_loop(filename_image)
  
  os.makedirs('./output/', exist_ok = True)
  
  for filename, image in result.items(): 
      image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
      cv2.imwrite('./output/' + filename, image)
  
  response = {
    'inference_time': inference_time, 
    'number_of_images': len(filename_image), 
    'average_inference_time': inference_time/len(filename_image)
  }
  status_code = Response(response = str(response), status = 200)
  return status_code

#routing http posts to this method
@app.route('/api/detect/image', methods=['POST', 'GET'])
def detect():
    image = request.files["images"]
    pil_image = Image.open(image.stream).convert('RGB')
    image_name = image.filename

    pil_image = ImageOps.fit(pil_image, (512, 512), Image.ANTIALIAS)
    filename_image = {image_name: pil_image}

    
    result, inference_time = detection_loop(filename_image)

    image = result[image_name]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.jpg', image)
    response = img_encoded.tostring()
    
    try:
        return Response(response=response, status=200, mimetype='image/jpg')
    except FileNotFoundError:
        abort(404)

#routing http posts to this method
@app.route('/api/detect/json', methods=['POST'])
def detect2():
    data_input = request.values.get('input')

    path = data_input
    filename_image = {}
    
    input_format = ["jpg", "png", "jpeg"]
    if data_input.find(".") != -1:
        print(data_input + " is a file")
        split_data_input = data_input.split(".", 1)
        if data_input.endswith(tuple(input_format)):
            print("INPUT FORMAT: %s IS VALID" % split_data_input[1])
            path_splitted = re.split('/', data_input)
            filename = path_splitted[-1]
            filename_image[filename] = Image.open(data_input).convert('RGB')
    else:
        print(data_input + " is a path with the following files: ")
        for filename in os.listdir(data_input):
            image_path = data_input + filename
            filename_image[filename] = Image.open(image_path).convert('RGB')
            print("  " + filename)
  
    result, inference_time = detection_loop(filename_image, as_json = True)
    response = {
        'inference_time': inference_time, 
        'number_of_images': len(filename_image), 
        'average_inference_time': inference_time/len(filename_image),
        'result': result
    }

    status_code = Response(response = json.dumps(response, cls=NumpyEncoder), status = 200)
    return status_code


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
