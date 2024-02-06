import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import sys
import os

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

image_directory = 'path/to/image/directory'
image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

output_directory = '/inference_output/'
os.makedirs(output_directory, exist_ok=True)

tflite_model = 'tflite_files/mob_ssd_v2.tflite'

path2label_map = 'label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(path2label_map, use_display_name=True)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

# Loading the TFLite model
tflite_interpreter = tf.lite.Interpreter(model_path=tflite_model)
tflite_interpreter.allocate_tensors()
tflite_input_details = tflite_interpreter.get_input_details()[0]
tflite_scores = tflite_interpreter.get_output_details()[0]
tflite_classes = tflite_interpreter.get_output_details()[3]
tflite_detections = tflite_interpreter.get_output_details()[2]
tflite_boxes = tflite_interpreter.get_output_details()[1]

# Loop through the list of image paths and perform inference
for image_path in image_paths:
    image_np = load_image_into_numpy_array(image_path)
    image_resize = tf.image.resize(image_np, [320, 320])
    normalize_img = (image_resize - 127.5) / 127.5
    tflite_input_tensor = np.expand_dims(normalize_img, 0)

    tflite_interpreter.set_tensor(tflite_input_details['index'], tflite_input_tensor)
    tflite_interpreter.invoke()

    tflite_detection_scores = tflite_interpreter.get_tensor(tflite_scores["index"])

    output_scale, output_zero_point = tflite_scores["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        print("need output quantization",output_scale,output_zero_point)
        tflite_detection_scores = tflite_detection_scores.astype(np.float32)
        tflite_detection_scores = (tflite_detection_scores - output_zero_point) * output_scale

    tflite_detection_classes = tflite_interpreter.get_tensor(tflite_classes["index"])
    output_scale, output_zero_point = tflite_classes["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        print("need output quantization",output_scale,output_zero_point)
        tflite_detection_classes = tflite_detection_classes.astype(np.float32)
        tflite_detection_classes = (tflite_detection_classes - output_zero_point) * output_scale
    tflite_num_detections = tflite_interpreter.get_tensor(tflite_detections["index"])
    output_scale, output_zero_point = tflite_detections["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        print("need output quantization",output_scale,output_zero_point)
        tflite_num_detections = tflite_num_detections.astype(np.float32)
        tflite_num_detections = (tflite_num_detections - output_zero_point) * output_scale
    tflite_input_tensor = np.expand_dims(normalize_img, 0)

    tflite_detection_boxes = tflite_interpreter.get_tensor(tflite_boxes["index"])
    output_scale, output_zero_point = tflite_boxes["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        print("need output quantization",output_scale,output_zero_point)
        tflite_detection_boxes = tflite_detection_boxes.astype(np.float32)
        tflite_detection_boxes = (tflite_detection_boxes - output_zero_point) * output_scale

    tflite_detection_scores = tflite_interpreter.get_tensor(tflite_scores["index"])
    tflite_detection_classes = tflite_interpreter.get_tensor(tflite_classes["index"])
    tflite_num_detections = tflite_interpreter.get_tensor(tflite_detections["index"])
    tflite_detection_boxes = tflite_interpreter.get_tensor(tflite_boxes["index"])

    print("*******************************************************")
    tflite_detection_scores = tflite_detection_scores[0]
    print(tflite_detection_scores)
    tflite_detection_classes = tflite_detection_classes[0].astype(np.int64)
    tflite_num_detections = tflite_num_detections[0].astype(np.int64)
    print(tflite_num_detections)
    tflite_detection_boxes = tflite_detection_boxes[0]
    print(tflite_detection_boxes)
    print("*******************************************************")

    tflite_image_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        tflite_image_detections,
        tflite_detection_boxes,
        tflite_detection_classes + 1,
        tflite_detection_scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.5,
        agnostic_mode=False,
        line_thickness=5
    )

    output_image_path = os.path.join(output_directory, os.path.basename(image_path))
    Image.fromarray(tflite_image_detections).save(output_image_path)
