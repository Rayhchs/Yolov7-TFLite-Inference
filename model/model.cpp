#include "model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

using namespace std;
using namespace cv;

// YOLO Detection
void Detection::init_det(std::string model_dir)
{

    // Load model
    model = tflite::FlatBufferModel::BuildFromFile(model_dir.c_str());

    if (!model)
    {
        std::cout << "Invalid to load recognition model!" << std::endl;
        return;
    }

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter)
    {
        std::cout << "Invalid to construct interpreter! (Recognition model)" << std::endl;
        return;
    }

    if(true)
    {
      interpreter->SetNumThreads(6);
    }

    // resize tensors
    if(interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cout << "Invalid to allocate tensors! (Recognition model)" << std::endl;
    }

    // Inference threshold
    box_threshold = 0.35;
    nms_threshold = 0.45;

    // get input(0) size
    int input = interpreter->inputs()[0];
    TfLiteIntArray* input_dims = interpreter->tensor(input)->dims;

    i_height = input_dims->data[1];   // height
    i_width = input_dims->data[2];    // width
    i_channels = input_dims->data[3]; // channels

    // get output(0) size
    int output = interpreter->outputs()[0];
    int output2 = interpreter->outputs()[1];
    int output3 = interpreter->outputs()[2];

    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    fmap_num = output_dims->data[1];
    nout = output_dims->data[2];
}

void Detection::inference_det(Mat &img_in, std::vector<PredBox> &feat_list, YAML::Node coco)
{
    // check image source
    Mat img_raw = img_in;
    cv::cvtColor(img_in, img_raw, cv::COLOR_BGR2RGB);
    if (img_in.empty() || img_in.channels() != 3){
        std::cout << "Invalid Input" << std::endl;
        return;
    }

    // resize the face_in size according to NN input size.
    Mat img_resize;
    resize(img_in, img_resize, Size(640, 640), 0, 0, INTER_LINEAR);

    // Input tensor
    std::vector<float> in;
    in.resize(img_resize.rows * img_resize.cols * img_resize.channels());
    std::vector<float> dst_data;
    std::vector<Mat> bgrChannels(3);
    split(img_resize, bgrChannels);
    for(auto i = 0; i < bgrChannels.size();i++){
        std::vector<float> data_ = std::vector<float>(bgrChannels[i].reshape(1,1));
        dst_data.insert(dst_data.end(),data_.begin(),data_.end());
    }

    // fill data to tensor input(0)
    for(size_t i = 0; i < in.size(); i++)
    {
        interpreter->typed_input_tensor<float>(0)[i] = (dst_data[i]/255.0);
    }

    // Invoke
    int ret = interpreter->Invoke();

    if(ret)
    {
        return;
    }

	vector<PredBox> bbox_collection;
	float ratioh = (float)img_in.rows / 640, ratiow = (float)img_in.cols / 640;
    float* pred = (float*)interpreter->typed_output_tensor<float>(0);

    // Inference stage
    for (int n = 0; n < fmap_num; n++)
    {
        float box_score = pred[4];
        if (box_score > box_threshold)
        {
            int max_ind = 0;
            float max_class_socre = 0;
            for (int k = 0; k < 80; k++)
            {
                if (pred[k + 5] > max_class_socre)
                {
                    max_class_socre = pred[k + 5];
                    max_ind = k;
                }
            }
            max_class_socre *= box_score;
            if (max_class_socre > box_threshold)
            { 
                float cx = pred[0] * ratiow ;
                float cy = pred[1] * ratioh;
                float w = pred[2] * ratiow;
                float h = pred[3] * ratioh;

                float xmin = cx - 0.5 * w;
                float ymin = cy - 0.5 * h;
                float xmax = cx + 0.5 * w;
                float ymax = cy + 0.5 * h;

                bbox_collection.push_back(PredBox{xmin, ymin, xmax, ymax, max_class_socre, max_ind});
            }
        }
        pred += nout;
    }
	
    NMS(bbox_collection);
	for (size_t i = 0; i < bbox_collection.size(); ++i)
	{
        feat_list.push_back(bbox_collection[i]);
		int xmin = int(bbox_collection[i].x1);
		int ymin = int(bbox_collection[i].y1);
		rectangle(img_raw, Point(xmin, ymin), Point(int(bbox_collection[i].x2), int(bbox_collection[i].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", bbox_collection[i].score);
        string name = coco["names"][bbox_collection[i].label].as<std::string>();
		label = name + ":" + label;
		putText(img_raw, label, Point(xmin, ymin - 1), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 1);
	}
}

void Detection::NMS(vector<PredBox>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](PredBox a, PredBox b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= nms_threshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const PredBox& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}