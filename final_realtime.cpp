/*
 * Date: 2022/10/26
 * Goal:
 * 1. Open video stream device and show on LCD
 * 2. Press "c" and screenshot, save to file
 * 3. Open executable and auto record to an avi file
 */

#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/mman.h> 
using namespace std;

struct framebuffer_info {
  uint32_t bits_per_pixel;
  uint32_t xres_virtual;
  uint32_t yres_virtual;
};

int screenshot = 0;
int release = 0;
pthread_mutex_t mutex;

struct framebuffer_info
get_framebuffer_info(const char *framebuffer_device_path);

void *readval(void *ptr) {
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  char c;
  while (true) {
    if (read(STDIN_FILENO, &c, 1) == 1) {
      // reset the input buffer c
      if (c == 'q') {
        release = 1;
        break;
      }
    }
  }
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

void *camera_func(void *ptr) {
  cv::Mat frame, dst;
  cv::Size frame_size;
  cv::VideoCapture camera(2);

  framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
  int fb_size = fb_info.yres_virtual * fb_info.xres_virtual * 2;

  int fb_fd = open("/dev/fb0", O_RDWR);
  if (fb_fd == -1) {
    perror("Error opening framebuffer device");
    exit(EXIT_FAILURE);
  }

  char *fb_ptr = (char *)mmap(NULL, fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
  if (fb_ptr == MAP_FAILED) {
    perror("Error mapping framebuffer device to memory");
    close(fb_fd);
    exit(EXIT_FAILURE);
  }

  camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, 960);
  camera.set(cv::CAP_PROP_FPS, 30);
  int cnt = 0;
  char *filename = new char[100];

  while (true) {
    camera >> frame;
    frame_size = frame.size();
    cv::cvtColor(frame, dst, cv::COLOR_BGR2BGR565);


    // The screen resolution is 800x480
    // The frame resolution is 640x480
    // centered the video and left 80 on left and 80 on right
    int center_x = (fb_info.xres_virtual - frame_size.width) / 2;
    int center_y = (fb_info.yres_virtual - frame_size.height) / 2;
    for (int y = 0; y < frame_size.height; y++) {
      char *fb_offset = fb_ptr + (((y+center_y) * fb_info.xres_virtual) + center_x) * 2;
      memcpy(fb_offset, dst.ptr(y), frame_size.width * 2);
    }

    if (release) {
      break;
    }
  }

  munmap(fb_ptr, fb_size);
  close(fb_fd);
  camera.release();
}

int main(int argc, const char *argv[]) {

  // create a thread spciified to read val screenshot
  pthread_t thread1;
  pthread_t thread2;
  pthread_mutex_init(&mutex, NULL);

  // read the input from keyboard
  pthread_create(&thread1, NULL, &readval, NULL);
  // read from camera and show on framebuffer, move this to thread2
  pthread_create(&thread2, NULL, &camera_func, NULL);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  pthread_mutex_destroy(&mutex);
  return 0;
}

struct framebuffer_info
get_framebuffer_info(const char *framebuffer_device_path) {
  struct framebuffer_info fb_info;
  struct fb_var_screeninfo screen_info;
  int fd = open(framebuffer_device_path, O_RDWR);
  int attr = ioctl(fd, FBIOGET_VSCREENINFO, &screen_info);

  fb_info.xres_virtual = screen_info.xres;
  fb_info.yres_virtual = screen_info.yres;
  fb_info.bits_per_pixel = screen_info.bits_per_pixel;

  return fb_info;
};
