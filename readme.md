# Yolov7-TFLite-Inference C++
This repository implment Inference of YOLOv7 TFLite in C++. To start this program, please download [yolov7.pt](https://github.com/WongKinYiu/yolov7) model first. This model has already concatenated the output shape with (1, n, 85) which n is based on different input size.

## Usage

* Download this repository
``` shell

git clone https://github.com/Rayhchs/Yolov7-TFLite-Inference.git

```

* Download Yolov7 onnx model from https://github.com/WongKinYiu/yolov7 and convert yolov7.pt to yolov7.onnx
``` shell

git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
python export.py --grid

```
After conversion, put the model to **./onnx2tflite**

* Convert onnx to tflite
``` shell

cd onnx2tflite
python onnx2tflite

```
Afterward, put the tflite model to **./weight**

* Compile and run
``` shell

make
./main

```

## Test result

* Input
<img src="https://github.com/Rayhchs/Yolov7-TFLite-Inference/blob/main/dataset/test.jpg" width="256"/>

* Result
<img src="https://github.com/Rayhchs/Yolov7-TFLite-Inference/blob/main/result/output.jpg" width="256"/>