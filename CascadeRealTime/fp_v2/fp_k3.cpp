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
#include <pthread.h>


#define pi 3.1415926
volatile sig_atomic_t gSignalStatus = 0;
cv::VideoCapture camera(2);
cv::Mat frame;
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

struct ThreadArgs { 
    framebuffer_info fb_info;
	int blue;
	int yellow;
	int red;
};


int min(int a, int b){
	if(a<b)return a;
	else return b;
}

int max(int a, int b){
	if(a>b)return a;
	else return b;
}

//画点函数 
void draw_dot(fbdev dev, int x, int y, int blue, int yellow, int red){
	int xres=dev.xres_virtual;
	int yres=dev.yres_virtual;
	int bpp=dev.bits_per_pixel;
	long offset=(y*xres+x)*bpp/8;
	*(dev.fbp+offset)=blue; //设置颜色，默认为白色 
	*(dev.fbp+offset+1)=yellow;
	*(dev.fbp+offset+2)=red;
}

//画线函数 
void draw_line(fbdev dev, int x1,int y1, int x2, int y2 ,int blue, int yellow, int red){ //,blue,yellow,red
	int i,j;
	if(x1==x2){
		for(j=min(y1,y2);j<=max(y1,y2);j++)
			draw_dot(dev,x1,j,blue,yellow,red);
		return;
	}
	if(y1==y2){
		for(i=min(x1,x2);i<=max(x1,x2);i++)
			draw_dot(dev,i,y1,blue,yellow,red);
		return;
	}
	if(x1<x2){
		for(i=x1;i<x2;i++){
			if(y1<y2){
				for(j=y1+(i-x1)*(y2-y1)/(x2-x1);j<=y1+(i+1-x1)*(y2-y1)/(x2-x1);j++)
					draw_dot(dev,i,j,blue,yellow,red);
			}
			else {
				for(j=y1+(i-x1)*(y2-y1)/(x2-x1);j>=y1+(i+1-x1)*(y2-y1)/(x2-x1);j--)
					draw_dot(dev,i,j,blue,yellow,red);
			}
		}
	}
	else{
        for(i=x1;i>x2;i--){
            if(y1<y2){
                for(j=y1+(x1-i)*(y2-y1)/(x1-x2);j<=y1+(x1-1-i)*(y2-y1)/(x1-x2);j++)
                    draw_dot(dev,i,j,blue,yellow,red);
            }
        	else {
                for(j=y1+(x1-i)*(y2-y1)/(x1-x2);j>=y1+(x1-1-i)*(y2-y1)/(x1-x2);j--)
                	draw_dot(dev,i,j,blue,yellow,red);
			}
		}
	}					
}

//画圆函数 
/*
void draw_circle(fbdev dev, int x, int y, int r){
	int i,a,b;
	for(i=0;i<360;i++){
		a=x+r*cos(i/180.0*pi);
		b=y+r*sin(i/180.0*pi);
		draw_dot(dev,a,b);
	}
}
*/

void *detect_screen(void *args){
	// 将参数结构体强制类型转换回原始类型

	struct ThreadArgs *threadArgs = (struct ThreadArgs *)args;


	fbdev dev = threadArgs->fb_info;
	Mat Image;
	int blue = threadArgs->blue;
	int yellow = threadArgs->yellow;
	int red = threadArgs->red;
	while(true){
		cvtColor(frame, Image, cv::COLOR_BGR2GRAY);
		blur(Image, Image, Size(5, 5));
		threshold(Image, Image, 60, 255, THRESH_BINARY);
		cv::bitwise_not(Image, Image);
		Mat labels, stats, centroids;
		int num_labels = connectedComponentsWithStats(Image, labels, stats, centroids, 8);
		//std::cout << "Number of objects: " << num_labels - 1;  

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
				//std::cout<<"screen"<<stats.at<int>(max_idx, i)<<" ";
			}
			//std::cout<<"\n";
			int x = stats.at<int>(max_idx, 0);
			int y = stats.at<int>(max_idx, 1);
			int w = stats.at<int>(max_idx, 2);
			int h = stats.at<int>(max_idx, 3);
			//std::cout << x << " " <<  y << " " << w << " " << h << "\n";
		   
			draw_line(dev,x,y,x+w,y,blue,yellow,red);
			draw_line(dev,x,y,x,y+h,blue,yellow,red);
			draw_line(dev,x,y+h,x+w,y+h,blue,yellow,red);
			draw_line(dev,x+w,y,x+w,y+h,blue,yellow,red);
		}
	}
   
}

void *detect_keyboard(void *args){
	
	// 将参数结构体强制类型转换回原始类型
    struct ThreadArgs *threadArgs = (struct ThreadArgs *)args;

	fbdev dev = threadArgs->fb_info;
	int blue = threadArgs->blue;
	int yellow = threadArgs->yellow;
	int red = threadArgs->red;

	// Detect keyboard Convert the image to grayscale (required for object detection)
    cv::Mat gray;

	while(true){
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		// Load the pre-trained Haar Cascade XML file
		cv::CascadeClassifier cascade;
		if (!cascade.load("./cascade_k3.xml"))
		{
		    std::cerr << "Error loading cascade file!" << std::endl;
		    
		}
		// Detect objects in the image
		std::vector<cv::Rect> detections;
		// image, obj, numDetections, scaleFactor, minNeighbors, flag, max_min size
		cascade.detectMultiScale(gray, detections, 1.1, 1, 0, cv::Size(75, 75));

		for (int box = 0; box < detections.size(); box++){
			int x = detections[box].x;
			int y = detections[box].y;
			int w = detections[box].width;
			int h = detections[box].height;
			std::cout << "keyboard" << x << " " <<  y << " " << w << " " << h << "\n";
			draw_line(dev,x,y,x+w,y,blue,yellow,red);
			draw_line(dev,x,y,x,y+h,blue,yellow,red);
			draw_line(dev,x,y+h,x+w,y+h,blue,yellow,red);
			draw_line(dev,x+w,y,x+w,y+h,blue,yellow,red);
		}
	}

}

void *detect_mouse(void *args){
	// 将参数结构体强制类型转换回原始类型
    struct ThreadArgs *threadArgs = (struct ThreadArgs *)args;
	fbdev dev = threadArgs->fb_info;
	int blue = threadArgs->blue;
	int yellow = threadArgs->yellow;
	int red = threadArgs->red;


	// Detect keyboard Convert the image to grayscale (required for object detection)
    cv::Mat gray;
	while(true){
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// Load the pre-trained Haar Cascade XML file
		cv::CascadeClassifier cascade;
		if (!cascade.load("./cascade_m.xml"))
		{
		    std::cerr << "Error loading cascade file!" << std::endl;
		    
		}
		// Detect objects in the image
		std::vector<cv::Rect> detections;
		// image, obj, numDetections, scaleFactor, minNeighbors, flag, max_min size
		cascade.detectMultiScale(gray, detections, 1.1, 3, 0, cv::Size(80, 80));

		for (int box = 0; box < detections.size(); box++){
			int x = detections[box].x;
			int y = detections[box].y;
			int w = detections[box].width;
			int h = detections[box].height;
			std::cout << "mouse" << x << " " <<  y << " " << w << " " << h << "\n";
			draw_line(dev,x,y,x+w,y,blue,yellow,red);
			draw_line(dev,x,y,x,y+h,blue,yellow,red);
			draw_line(dev,x,y+h,x+w,y+h,blue,yellow,red);
			draw_line(dev,x+w,y,x+w,y+h,blue,yellow,red);
		}
	}
}



struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

int main(int argc, const char *argv[])
{
	std::signal(SIGINT, handleCtrlC);
	// void detect_screen(fbdev dev, Mat Image,int blue,int yellow,int red);
	// void detect_keyboard(fbdev dev, Mat frame,int blue,int yellow,int red);
	// void detect_mouse(fbdev dev, Mat frame,int blue,int yellow,int red);
    // Variable to store the frame get from video stream
    
    cv::Size2f frame_size;

    // Open video stream device
    

    // Get info of the framebuffer
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");

    // Open the framebuffer device
    std::ofstream ofs("/dev/fb0");
	
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

	// Create thread
	pthread_t t1,t2,t3;
	struct ThreadArgs *argst1 = (struct ThreadArgs *)malloc(sizeof(struct ThreadArgs));
	argst1->fb_info = fb_info;
	argst1->blue = 255;
	argst1->yellow = 0;
	argst1->red = 0;

	struct ThreadArgs *argst2 = (struct ThreadArgs *)malloc(sizeof(struct ThreadArgs));
	argst2->fb_info = fb_info;
	argst2->blue = 0;
	argst2->yellow = 255;
	argst2->red = 0;

	struct ThreadArgs *argst3 = (struct ThreadArgs *)malloc(sizeof(struct ThreadArgs));
	argst3->fb_info = fb_info;
	argst3->blue = 150;
	argst3->yellow = 150;
	argst3->red = 0;
	
	Mat sframe;
	int flag = 0;
    while (true)
    {
        // Get video frame from stream
        camera >> frame;
		
		// Detect object
		/*detect_screen(fb_info,frame,255,0,0);
		detect_keyboard(fb_info,frame,0,255,0);
		detect_mouse(fb_info,frame,0,0,255);*/
        // Get size of the video frame
        frame_size = frame.size();

        
		
        
		
        // Transfer color space from BGR to BGR565 (16-bit image) to fit the requirement of the LCD
		cv::cvtColor(frame, sframe, cv::COLOR_BGR2BGR565);
        for (int y = 0; y < frame_size.height; y++)
        {
            int position = y * fb_info.xres_virtual * 2;
            ofs.seekp(position);
            ofs.write(reinterpret_cast<char *>(sframe.ptr(y)), frame_size.width * 2);
        }
		if (flag==0){		
				if (pthread_create(&t1, NULL, detect_screen, (void *)argst1) != 0) {
				cerr << "Error: pthreadt1_create\n";
				}
				if (pthread_create(&t2, NULL, detect_keyboard, (void *)argst2) != 0) {
					cerr << "Error: pthreadt2_create\n";
				}
				if (pthread_create(&t3, NULL, detect_mouse, (void *)argst3) != 0) {
					cerr << "Error: pthreadt3_create\n";
				}

				pthread_detach(t1);
				pthread_detach(t2);
				pthread_detach(t3);
			}
			flag = 1;
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



