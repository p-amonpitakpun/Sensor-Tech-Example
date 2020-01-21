// PingPong01.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <string.h>

using namespace cv;

const double gamma = 1 / 2.2;

Scalar lowerBound = Scalar(100, 100, 100);
Scalar upperBound = Scalar(255, 255, 200);

Mat gammaTable(1, 256, CV_8U);

Mat preprocess(Mat& src);
Mat ppDetect(Mat& src);
Mat postprocess(Mat& frame, Mat& image, Mat& detect);

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
	Mat detect;
	Mat postImage;

	String wName = "videocapture";
	int timeout = 30;

	namedWindow(wName, 1);

	double gamma = 0.67;
	uchar* p = gammaTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	// setup end

	for (;;) {
		// loop begin
		// get a new frame from camera
		cap >> frame;

		image = preprocess(frame);

		detect= ppDetect(image);

		postImage = postprocess(frame, image, detect);

		imshow(wName, postImage);

		// key check with timeout
		if (waitKey(timeout) >= 0) break;
		if (getWindowProperty(wName, WND_PROP_VISIBLE) == 0) break;
		// loop end
	}

	return 0;
}

Mat preprocess(Mat& src)
{
	Mat hsv;
	Mat mask;

	// gamma correction
	//CV_Assert(gamma >= 0);
	//Mat src_corrected = hsv.clone();
	//LUT(src, gammaTable, src_corrected);

	cvtColor(src, hsv, COLOR_BGR2HSV);
	//cvtColor(src_corrected, hsv, COLOR_BGR2HSV);


	return hsv;
}

Mat ppDetect(Mat& src)
{
	Mat mask;
	Mat mask_erode;
	Mat dst;

	// color detection
	inRange(src, lowerBound, upperBound, mask);

	// erosion
	int erosion_size = 1;
	Mat element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	erode(mask, mask_erode, element);

	return mask_erode;
}

Mat postprocess(Mat& frame, Mat& image, Mat& detect)
{
	Mat detect_bgr;
	Mat dst;

	cvtColor(detect, detect_bgr, COLOR_GRAY2BGR);
	hconcat(frame, detect_bgr, dst);

	return dst;
}