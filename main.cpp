#include "model/model.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using namespace std;
using namespace cv;

int main()
{
    // Model Initialization
    Detection det;
    vector<PredBox> out_feature;
    det.init_det("./weight/yolov7.tflite");
    Mat img = cv::imread("./dataset/test.jpg");
    YAML::Node coco = YAML::LoadFile("coco.yaml");
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Inference
    det.inference_det(img, out_feature, coco);
    
    // Write
    cv::imwrite("./result/output.jpg", img);
    ofstream out;
    out.open("./result/output.txt");
    if (out.fail()){
        cout << "input file opening failed...";
        exit(1);
    }
    out << "x1, y1, x2, y2, label, score" << endl;
    for (int i=0; i<out_feature.size(); i++){
        string x1 = to_string(out_feature[i].x1);
        string y1 = to_string(out_feature[i].y1);
        string x2 = to_string(out_feature[i].x2);
        string y2 = to_string(out_feature[i].y2);
        string label = coco["names"][out_feature[i].label].as<std::string>();
        string score = to_string(out_feature[i].score);
        string output = x1 + ", " + y1 + ", " + x2 + ", " + y2 + ", " + label + ", " + score;
        out << output << endl;
    }

    return 0;
}
