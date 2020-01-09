// PingPong01.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <iostream>
#include <string.h>

using namespace cv;

Mat gammaCorrection(const Mat& img, Mat& lookUpTable, const double gamma_);

int main()
{
	// Mat image = Mat::zeros(300, 600, CV_8UC3);
	// circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
	// circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
	// imshow("Display Window", image);
	// waitKey(0);

	// setup begin
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cout << "Video Capture is not opened." << std::endl;
		return 1;
	}

	Mat frame;
    Mat image;
	Mat hsv_image;
	Mat blues;

	String wName = "videocapture";
	int timeout = 30;

	namedWindow(wName, 1);

	double gamma = 0.67;
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	// setup end

	for (;;) {
		// loop begin
		// get a new frame from camera
		cap >> frame;

		// find a blue circle begin
		cvtColor(frame, hsv_image, COLOR_BGR2HSV);
		inRange(hsv_image, Scalar(100, 100, 100), Scalar(255, 255, 200), blues);
		// find a blue circle end

		// show the frame
		//imshow(wName, blues);
		//Mat res = image.clone();
		//LUT(image, lookUpTable, res);
        //imshow(wName, image);
		Mat image = gammaCorrection(frame, lookUpTable, 1/2.2);

		Mat img_gamma_corrected;

		hconcat(frame, image, img_gamma_corrected);
		imshow(wName, img_gamma_corrected);

		// key check with timeout
		if (waitKey(timeout) >= 0) break;

		if (getWindowProperty(wName, WND_PROP_VISIBLE) == 0) break;
		// loop end
	}

	return 0;
}

Mat gammaCorrection(const Mat& img, Mat& lookUpTable, const double gamma_)
{
	CV_Assert(gamma_ >= 0);

	Mat res = img.clone();

	LUT(img, lookUpTable, res);

	return res;
}
