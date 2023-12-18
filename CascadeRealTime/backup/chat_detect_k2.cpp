#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
int main() {
    // Load the pre-trained Haar Cascade XML file
    cv::CascadeClassifier cascade;
    if (!cascade.load("/Users/cheng/Desktop/key/result/cascade.xml")) {
        std::cerr << "Error loading cascade file!" << std::endl;
        return -1;
    }

    // Read the input image
    cv::Mat image = cv::imread("/Users/cheng/Desktop/mouth/example001.png");
    if (image.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    // Convert the image to grayscale (required for object detection)
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur
    //cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    //threshold(gray, gray, 100, 255, THRESH_BINARY);

    // Apply Canny edge detection
    // cv::Mat edges;
    // cv::Canny(gray, edges, 50, 150);


    // Detect objects in the image
    std::vector<cv::Rect> detections;
    // image, obj, numDetections, scaleFactor, minNeighbors, flag, max_min size
    cascade.detectMultiScale(gray, detections, 1.1, 1, 0, cv::Size(70, 70));

    // Draw rectangles around the detected objects
    for (const auto& rect : detections) {
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
    }

    // Display the result
    cv::imshow("Object Detection", image);
    cv::waitKey(0);

    return 0;
}
