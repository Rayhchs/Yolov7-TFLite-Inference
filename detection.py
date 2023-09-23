import tensorflow as tf
import cv2
import numpy as np

det_model_path = './yolov7.tflite'
interpreter = tf.lite.Interpreter(model_path = det_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_raw = cv2.imread('test.png')
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
img = cv2.resize(img_raw, (640, 640)).astype(np.float32)
img = img.transpose(2, 0, 1)
img =np.expand_dims(img,axis=0)
print(img.shape)
img = img/255.0

interpreter.set_tensor(input_details[0]['index'], img)  
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print(output.shape)
print(output[0,0,0:5])
