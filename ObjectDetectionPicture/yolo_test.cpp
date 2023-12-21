#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <stdio.h>
#include <algorithm>
#include <fstream>

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("yolov4/embedded_final.names");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

// Function to apply Non-maximum Suppression (NMS)
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, const std::vector<std::string>& class_names, float confThreshold = 0.5, float nmsThreshold = 0.4) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const cv::Mat& out : outs) {
        for (int i = 0; i < out.rows; ++i) {
            cv::Mat row = out.row(i);
            cv::Mat scores = row.colRange(5, out.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confThreshold) {
                int classId = classIdPoint.x;
                int centerX = static_cast<int>(row.at<float>(0) * frame.cols);
                int centerY = static_cast<int>(row.at<float>(1) * frame.rows);
                int width = static_cast<int>(row.at<float>(2) * frame.cols);
                int height = static_cast<int>(row.at<float>(3) * frame.rows);

                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classId);
                confidences.push_back(static_cast<float>(confidence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Apply Non-maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        // Draw bounding box and label
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("%s: %.2f", class_names[classIds[idx]].c_str(), confidences[idx]);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(box.y, labelSize.height);
        cv::putText(frame, label, cv::Point(box.x, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

int main() {
    // Load YOLOv5 model
    cv::dnn::Net net = cv::dnn::readNet("yolov4/yolov4-tiny-embedded_final_final.weights", "yolov4/yolov4-tiny-embedded_final.cfg", "yolov4/embedded_final.names");

    // Check if the model is loaded successfully
    if (net.empty()) {
        std::cerr << "Error loading YOLO model." << std::endl;
        return -1;
    }

    // Load image
    cv::Mat image = cv::imread("example001.png");
    if (image.empty()) {
        std::cerr << "Error loading image." << std::endl;
        return -1;
    }

    // Prepare input blob
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // Forward pass
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    // Apply NMS and draw bounding boxes
    std::vector<std::string> class_list = load_class_list();
    postprocess(image, outs, class_list, 0.5, 0.4);

    // Save the output image
    cv::imwrite("image_with_detections.png", image);
    std::cout << "finished by user\n";

    return 0;
}
