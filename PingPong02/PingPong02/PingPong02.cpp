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


int timeout = 25;
int erosion_size = 3;
int dilation_size = 1;

double thresh = 500;
double circularity = 0.0;
double cr = 0.00003;

Scalar lowerBound;
Scalar upperBound;

Mat gammaTable(1, 256, CV_8U);

void findMinMax(Mat& src, Mat& mask, Scalar& min, Scalar& max);
Mat process(Mat& frame);

int main(int argc, char** argv)
{
	// setup begin
	lowerBound = Scalar(65, 0, 160);
	upperBound = Scalar(180, 40, 235);

	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened()) {
		printf(">> webcam error\r\n");
		return 1;
	}

	Mat raw;
	Mat image;
	Mat detect;
	Mat postImage;

	String wName = "videocapture";

	namedWindow(wName, 1);

	for (;;) {
		// loop begin
		Mat frame;
		bool captured = cap.read(frame);

		if (captured) {
			imshow(wName, process(frame));
		}
		else {
			printf(">> ERROR : can't read the frame !");
		}

		// key check with timeout
		if (waitKey(timeout) >= 0) break;
		if (getWindowProperty(wName, WND_PROP_VISIBLE) == 0) break;
		// loop end
	}

	cap.release();
	return 0;
}

void findMinMax(Mat& src, Mat& mask, Scalar& min, Scalar& max)
{
	int nChannels = src.channels();

	Mat* channels = new Mat[nChannels];
	split(src, channels);

	for (int i = 0; i < nChannels; i++) {
		minMaxLoc(channels[i], &min[i], &max[i], NULL, NULL, mask);
	}
	return;
}

static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

Mat process(Mat& frame)
{
	Mat gblur;

	GaussianBlur(frame, gblur, Size(9, 9), 5, 0);

	Mat hsv;
	cvtColor(gblur, hsv, COLOR_BGR2HSV);

	Mat gray;
	cvtColor(gblur, gray, COLOR_BGR2GRAY);

	// find edges
	Mat canny;
	Canny(gray, canny, 0, thresh, 5);
	dilate(canny, canny, Mat(), Point(-1, -1));

	// Find contours
	std::vector<std::vector<Point>> contours;
	findContours(canny, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	Mat draw = frame.clone();
	std::vector<Point> approx;

	for (size_t i = 0; i < contours.size(); i++) {
		approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
		if (approx.size() == 4 &&
			fabs(contourArea(approx)) > 1000 &&
			isContourConvex(approx))
		{
			Rect boundRect = boundingRect(approx); 
			Point center(boundRect.x + 0.5 * boundRect.width, boundRect.y + 0.5 * boundRect.height);
			Vec3b rectColor = frame.at<Vec3b>(center);
			rectangle(draw, boundRect, rectColor, -1);

			//polylines(draw, approx, true, Scalar(255, 0, 0), 3);

			putText(draw, "RECT", center, FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255), 2);
		}
	}

	Mat tmp1, tmp2;
	hconcat(frame, draw, tmp2);
	return tmp2;
}