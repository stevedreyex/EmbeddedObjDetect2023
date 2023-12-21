#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <linux/fb.h>
#include <sys/ioctl.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

std::string keys =
    "{ help  h     | | Print help message. }"
    "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes. }"
    "{ backend     | 0 | Choose one of computation backends: "
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU }";

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;


Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name) 
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes; 
    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) 
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) 
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) 
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    const std::string modelName = parser.get<String>("@alias");
    const std::string zooFile = parser.get<String>("zoo");

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run classification deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    /* Now we have only 540 x 540 onnx model
     * Expect 1280 x 960 as the examples001.jpg
     */

    // int inpWidth = parser.get<int>("width");
    int inpWidth = 540;
    // int inpHeight = parser.get<int>("height");
    int inpHeight = 540;
    // String model = findFile(parser.get<String>("model"));
    String config = findFile(parser.get<String>("config"));
    String framework = parser.get<String>("framework");
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    // CV_Assert(!model.empty());

    //! [Read and initialize network]
    Net net = readNet("models/yolov5s.onnx");
    //! [Read and initialize network]

    // Create a window
    // static const std::string kWinName = "Deep learning image classification in OpenCV";
    // namedWindow(kWinName, WINDOW_NORMAL);

    /*
     * Start the process using code from lab2 and lab3
     * Lab2 read the input image 
     * Lab3 write the optput to jpeg
     */
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");
    //! [Open a video file or an image file or a camera stream]
    // VideoCapture cap;
    // if (parser.has("input"))
    //     cap.open(parser.get<String>("input"));
    // else
    //     cap.open(0);
    //! [Open a video file or an image file or a camera stream]

    // Open a image file instead
    Mat frame, blob;
    frame = imread("examples001.jpg");

    // Process frames.

    // cap >> frame;
    // if (frame.empty())
    // {
    //     waitKey();
    //     break;
    // }


    // Refrence https://github.com/spmallick/learnopencv/blob/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python/yolov5.cpp
    // Preprocess starts here
    //! [Create a 4D blob from a frame]
    cv::cvtColor(frame, frame, cv::COLOR_BGR2BGR565);
    blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false);
    //! [Create a 4D blob from a frame]

    //! [Set input blob]
    net.setInput(blob);
    //! [Set input blob]
    //! [Make forward pass]
    vector<Mat> prob;
    net.forward(prob, net.getUnconnectedOutLayersNames());
    //! [Make forward pass]

    // Show the lables on image: class name and confidence
    // we have class names vector of string
    Mat img = post_process(frame.clone(), prob, classes);

    // imshow(kWinName, frame);
    sprintf(filename, "/run/media/mmcblk1p1/screenshot/image_%d.png", cnt++);
    cout << "save to " << filename << endl;
    cv::imwrite(filename, frame);

    return 0;
}
