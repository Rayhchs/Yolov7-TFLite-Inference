#define MODEL_H

#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/kernels/register.h>

using namespace std;
using namespace cv;

struct PredBox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
};

class Detection
{
private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    int i_height;
    int i_width;
    int i_channels;
    int fmap_num;
    int nout;
    float box_threshold;
    float nms_threshold;

public:
    void init_det(std::string model_dir);
    void inference_det(Mat &img_in, std::vector<PredBox> &feat_list, YAML::Node coco);
    void NMS(std::vector<PredBox> &input);
};