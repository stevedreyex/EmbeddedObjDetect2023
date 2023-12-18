#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#include <ctime>
#include <sstream>
#include <opencv2/opencv.hpp>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<sys/ioctl.h>
#include<linux/fb.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define pi 3.1415926
volatile sig_atomic_t gSignalStatus = 0;
cv::VideoCapture camera(2);
using namespace cv;
using namespace std;

void handleCtrlC(int signal) {
    std::cout << "Ctrl+C 按下，执行一些操作..." << std::endl;
    // 设置标志，表示已经接收到Ctrl+C信号
    gSignalStatus = 1;
	camera.release();
	exit(signal);

}





typedef struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixels in a row in virtual screen
	uint32_t yres_virtual;

	int fd;
	long size;
	unsigned char*fbp;
	struct fb_var_screeninfo vinfo;
	struct fb_fix_screeninfo finfo;
}fbdev;


int min(int a, int b){
	if(a<b)return a;
	else return b;
}

int max(int a, int b){
	if(a>b)return a;
	else return b;
}

//画点函数 
void draw_dot(fbdev dev, int x, int y){
	int xres=dev.xres_virtual;
	int yres=dev.yres_virtual;
	int bpp=dev.bits_per_pixel;
	long offset=(y*xres+x)*bpp/8;
	*(dev.fbp+offset)=0; //设置颜色，默认为白色 
	*(dev.fbp+offset+1)=255;
	*(dev.fbp+offset+2)=0;
}

//画线函数 
void draw_line(fbdev dev, int x1,int y1, int x2, int y2){
	int i,j;
	if(x1==x2){
		for(j=min(y1,y2);j<=max(y1,y2);j++)
			draw_dot(dev,x1,j);
		return;
	}
	if(y1==y2){
		for(i=min(x1,x2);i<=max(x1,x2);i++)
			draw_dot(dev,i,y1);
		return;
	}
	if(x1<x2){
		for(i=x1;i<x2;i++){
			if(y1<y2){
				for(j=y1+(i-x1)*(y2-y1)/(x2-x1);j<=y1+(i+1-x1)*(y2-y1)/(x2-x1);j++)
					draw_dot(dev,i,j);
			}
			else {
				for(j=y1+(i-x1)*(y2-y1)/(x2-x1);j>=y1+(i+1-x1)*(y2-y1)/(x2-x1);j--)
					draw_dot(dev,i,j);
			}
		}
	}
	else{
        for(i=x1;i>x2;i--){
            if(y1<y2){
                for(j=y1+(x1-i)*(y2-y1)/(x1-x2);j<=y1+(x1-1-i)*(y2-y1)/(x1-x2);j++)
                    draw_dot(dev,i,j);
            }
        	else {
                for(j=y1+(x1-i)*(y2-y1)/(x1-x2);j>=y1+(x1-1-i)*(y2-y1)/(x1-x2);j--)
                	draw_dot(dev,i,j);
			}
		}
	}					
}

//画圆函数 
void draw_circle(fbdev dev, int x, int y, int r){
	int i,a,b;
	for(i=0;i<360;i++){
		a=x+r*cos(i/180.0*pi);
		b=y+r*sin(i/180.0*pi);
		draw_dot(dev,a,b);
	}
}

void detect_screen(fbdev dev, Mat Image){

    cvtColor(Image, Image, COLOR_BGR2GRAY);
    blur(Image, Image, Size(5, 5));
    threshold(Image, Image, 60, 255, THRESH_BINARY);
    cv::bitwise_not(Image, Image);

    Mat labels, stats, centroids;
    int num_labels = connectedComponentsWithStats(Image, labels, stats, centroids, 8);

    // 输出检测到的对象数量
    std::cout << "Number of objects: " << num_labels - 1;  // 减1是因为背景也被算在内

    int max_region = 0;
    int max_idx = 1;
    for (int i = 1;  i < num_labels; i++){
        if (stats.at<int>(i, 4) > max_region){
            max_region = stats.at<int>(i, 4);
            max_idx = i;
        }
    }
   

    /*Mat output = Mat::zeros(Image.size(), CV_8UC3);
    Mat mask = labels == max_idx;
    output.setTo(Scalar(rand() % 255, rand() % 255, rand() % 255), mask);*/
    //rectangle(output, Rect(stats.at<int>(max_idx, 0), stats.at<int>(max_idx, 1), stats.at<int>(max_idx, 2), stats.at<int>(max_idx, 3)), Scalar(255, 0, 255), 1, 8, 0);//外接矩形

    if (num_labels - 1!=0){
		for (int i=0; i<5; i++){
		    std::cout<<"screen"<<stats.at<int>(max_idx, i)<<" ";
		}
		std::cout<<"\n";
		int x = stats.at<int>(max_idx, 0);
		int y = stats.at<int>(max_idx, 1);
		int w = stats.at<int>(max_idx, 2);
		int h = stats.at<int>(max_idx, 3);
		//std::cout << x << " " <<  y << " " << w << " " << h << "\n";
	   
		draw_line(dev,x,y,x+w,y);
		draw_line(dev,x,y,x,y+h);
		draw_line(dev,x,y+h,x+w,y+h);
		draw_line(dev,x+w,y,x+w,y+h);
	}
   
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

int main(int argc, const char *argv[])
{
	std::signal(SIGINT, handleCtrlC);
	void detect_screen(fbdev dev, Mat Image);
    // Variable to store the frame get from video stream
    cv::Mat frame;
    cv::Size2f frame_size;

    // Open video stream device
    

    // Get info of the framebuffer
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");

    // Open the framebuffer device
    std::ofstream ofs("/dev/fb0");

    // Load the pre-trained Haar Cascade XML file
    cv::CascadeClassifier cascade;
    if (!cascade.load("./cascade_m.xml"))
    {
        std::cerr << "Error loading cascade file!" << std::endl;
        return 1;
    }

    // Check if video stream device is opened successfully or not
    if (!ofs || !camera.isOpened())
    {
        std::cerr << "Could not open video device." << std::endl;
        return 1;
    }

    // Set property of the frame
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480); // Set your desired height here
    camera.set(cv::CAP_PROP_FPS, 60);
    int count = 0;
    char screenshot_name[100];

    while (true)
    {
        // Get video frame from stream
        camera >> frame;

		// Detect Screen
		detect_screen(fb_info,frame);
        // Get size of the video frame
        frame_size = frame.size();

        // Convert the image to grayscale (required for object detection)
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect objects in the image
        std::vector<cv::Rect> detections;
        // image, obj, numDetections, scaleFactor, minNeighbors, flag, max_min size
        //cascade.detectMultiScale(gray, detections, 1.1, 1, 0, cv::Size(100, 100));

        // Draw rectangles around the detected objects
        /*for (const auto &rect : detections)
        {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        }*/
		/*for (int box = 0; box < detections.size(); box++){
			int x = detections[box].x;
			int y = detections[box].y;
			int w = detections[box].width;
			int h = detections[box].height;
			std::cout << x << " " <<  y << " " << w << " " << h << "\n";
			draw_line(fb_info,x,y,x+w,y);
			draw_line(fb_info,x,y,x,y+h);
			draw_line(fb_info,x,y+h,x+w,y+h);
			draw_line(fb_info,x+w,y,x+w,y+h);
		}*/
        // Transfer color space from BGR to BGR565 (16-bit image) to fit the requirement of the LCD
		cv::cvtColor(frame, frame, cv::COLOR_BGR2BGR565);
        for (int y = 0; y < frame_size.height; y++)
        {
            int position = y * fb_info.xres_virtual * 2;
            ofs.seekp(position);
            ofs.write(reinterpret_cast<char *>(frame.ptr(y)), frame_size.width * 2);
        }
    }


    // Closing video stream
    camera.release();

    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.
    int fd = open(framebuffer_device_path, O_RDWR);
	fb_info.fd = fd;


    if (fd >= 0)
    {
        if (!ioctl(fd, FBIOGET_VSCREENINFO, &screen_info))
        {
            fb_info.xres_virtual = screen_info.xres_virtual;
			fb_info.yres_virtual = screen_info.yres_virtual;
            fb_info.bits_per_pixel = screen_info.bits_per_pixel;
			fb_info.size = fb_info.xres_virtual * fb_info.yres_virtual *  fb_info.bits_per_pixel / 8;
			fb_info.fbp=(unsigned char*)mmap(0,fb_info.size,PROT_READ|PROT_WRITE,MAP_SHARED,fb_info.fd,0);
			memset(fb_info.fbp,0,fb_info.size);
        }
    }
    return fb_info;
};



