from onnx_tf.backend import prepare
import onnx
import tensorflow as tf
import os

def main(model_name):

	os.mkdir('./tf_weight') if not os.path.exists('./tf_weight') else None

	onnx_model_path = model_name + '.onnx'
	tf_model_path = './tf_weight/'

	onnx_model = onnx.load(onnx_model_path)
	tf_rep = prepare(onnx_model)
	tf_rep.export_graph(tf_model_path)


	saved_model_dir = './tf_weight/'
	tflite_model_path = saved_model_dir + model_name + '.tflite'

	# Convert the model
	converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
	tflite_model = converter.convert()

	# Save the model
	with open(tflite_model_path, 'wb') as f:
	    f.write(tflite_model)

if __name__ == '__main__':
	main('yolov7')