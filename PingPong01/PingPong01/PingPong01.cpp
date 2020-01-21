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


int timeout = 1000 / 60;
double gamma = 0.6;
double bound_err = 20;
int erosion_size = 5;
int dilation_size = 1;
float minRadius = 10;

Scalar lowerBound;
Scalar upperBound;

Mat gammaTable(1, 256, CV_8U);

void setup_orange();
Mat preprocess(Mat& src);
Mat ppDetect(Mat& src);
Mat postprocess(Mat& frame, Mat& image, Mat& detect);

int main(int argc, char** argv)
{
	// setup begin

	setup_orange();

	VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cout << "Video Capture is not opened." << std::endl;
		return 1;
	}

	Mat raw;
	Mat frame;
	Mat image;
	Mat detect;
	Mat postImage;

	String wName = "videocapture";

	namedWindow(wName, 1);

	uchar* p = gammaTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	// setup end

	for (;;) {
		// loop begin
		// get a new frame from camera
		cap >> raw;
		flip(raw, frame, 1);

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

void setup_orange()
{
	Mat orange = imread("orange.png", IMREAD_COLOR);
	printf(">> orange !!! \n");
	printf("|  width \t = %d \n", orange.rows);
	printf("|  height \t = %d \n", orange.cols);
	printf("|  channels \t = %d \n", orange.channels());

	Mat orange_blur;
	GaussianBlur(orange, orange_blur, Size(15, 15), 45, 0);

	Mat orange_hsv;
	cvtColor(orange_blur, orange_hsv, COLOR_BGR2HSV);

	int nChannels = orange_hsv.channels();
	Mat* orangeChannels = new Mat[nChannels];
	split(orange_hsv, orangeChannels);

	double* x_min = new double[nChannels];
	double* x_max = new double[nChannels];

	printf("|  ----\n");
	printf("|  min \t\t max\n");
	for (int i = 0; i < nChannels; i++) {
		minMaxLoc(orangeChannels[i], &x_min[i], &x_max[i]);
		printf("|  %.2f \t %.2f\n", x_min[i], x_max[i]);
	}

	CV_Assert(nChannels == 3);
	lowerBound = Scalar(x_min[0] - bound_err,
		x_min[1] - bound_err,
		x_min[2] - bound_err);
	upperBound = Scalar(x_max[0] + bound_err,
		x_max[1] + bound_err,
		x_max[2] + bound_err);

	printf("\n\n\n");
}

Mat preprocess(Mat& src)
{
	Mat gblur;
	Mat mask;

	GaussianBlur(src, gblur, Size(5, 5), 125, 0);

	// gamma correction
	CV_Assert(gamma >= 0);
	Mat src_corrected = gblur.clone();
	LUT(gblur, gammaTable, src_corrected);

	Mat hsv;
	cvtColor(gblur, hsv, COLOR_BGR2HSV);
	return hsv;
}

Mat ppDetect(Mat& src)
{
	Mat mask;
	Mat dst;

	// color detection
	inRange(src, lowerBound, upperBound, mask);

	// erosion
	Mat mask_erode;
	Mat erode_element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	erode(mask, mask_erode, erode_element);

	// dilate
	Mat mask_dilate;
	Mat dilate_element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	dilate(mask_erode, mask_dilate, dilate_element);

	return mask_dilate;
}

Mat postprocess(Mat& frame, Mat& image, Mat& detect)
{
	Mat tmp1, tmp2;
	Mat detect_bgr;
	Mat draw = frame.clone();
	Mat dst;

	// find contours
	std::vector<std::vector<Point>> contours;
	findContours(detect, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// if there is any contour
	if (contours.size() > 0) {

		// find the largest contour
		std::vector<Point> maxContour = contours.at(0);
		double maxArea = contourArea(maxContour);
		for (int i = 0; i < contours.size(); i++) {
			std::vector<Point> contour = contours.at(i);
			double area = contourArea(contour);
			if (area >= maxArea) {
				maxContour = contour;
				maxArea = area;
			}
		}

		// find the min enclosing circle
		Point2f center;
		float radius;
		minEnclosingCircle(maxContour, center, radius);

		if (radius > minRadius)
			circle(draw, center, radius, Scalar(0, 0, 255), 5);
	}

	cvtColor(detect, detect_bgr, COLOR_GRAY2BGR);
	hconcat(frame, image, tmp1);
	hconcat(detect_bgr, draw,tmp2);
	vconcat(tmp1, tmp2, dst);

	return dst;
}