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

const size_t nColorList = 11;

struct colorStruct {
	char name[20];
	double color[3];
};

typedef colorStruct colorPoint;

colorPoint colorList[11] = {
	{
		"brown",
		{ 0.469525490 , 0.248035196 , 0.147159020 }
	},
	{
		"dark_blue",
		{ 0.084106176 , 0.191335098 , 0.565377745 }
	},
	{
		"dark_green",
		{ 0.075102353 , 0.200504902 , 0.124711863 }
	},
	{
		"forest_green",
		{ 0.109641471 , 0.348973039 , 0.202970784 }
	},
	{
		"light_blue",
		{ 0.146241078 , 0.585789216 , 0.779026765 }
	},
	{
		"light_green",
		{ 0.271230490 , 0.448678235 , 0.106501667 }
	},
	{
		"orange",
		{ 0.574475098 , 0.183433922 , 0.098825686 }
	},
	{
		"pink",
		{ 0.731312745 , 0.202826471 , 0.463220098 }
	},
	{
		"purple",
		{ 0.495675294 , 0.158975980 , 0.428593137 }
	},
	{
		"red",
		{ 0.694599608 , 0.158629804 , 0.144396569 }
	},
	{
		"yellow",
		{ 0.957700000 , 0.881482059 , 0.129532941 }
	}
};

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

double euDistant(Vec3b color, colorPoint cPoint) {
	double r = color[2];
	double g = color[1];
	double b = color[0];
	double rgb = r + g + b;
	double rn = r / rgb;
	double gn = g / rgb;
	double bn = b / rgb;

	double cnorm = (double) cPoint.color[0] + (double) cPoint.color[1] + (double) cPoint.color[2];

	double dr = (rn - cPoint.color[0] / cnorm);
	double dg = (gn - cPoint.color[1] / cnorm);
	double db = (bn - cPoint.color[2] / cnorm);

	return sqrt(dr * dr + dg * dg + db * db);
}

void detectColor(Vec3b color, char str[20])
{
	double minDist = euDistant(color, colorList[0]), dist;
	size_t index = 0;
	for (size_t i = 1; i < nColorList; i++) {
		dist = euDistant(color, colorList[i]);
		if (dist <= minDist) {
			minDist = dist;
			index = i;
		}
	}
	strcpy_s(str, 20, colorList[index].name);
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
			

			// find the average color of the rectangle
			size_t near = 5;
			if (min(boundRect.width, boundRect.height) < (near * 2)) break;

			double r = 0, g = 0, b = 0;
			for (size_t i = 0; i < (near * 2); i++) {
				for (size_t j = 0; j < (near * 2); j++) {
					Point point(center.x - near + i, center.y - near + i);
					Vec3b color = gblur.at<Vec3b>(point);
					r += color[2];
					g += color[1];
					b += color[0];
				}
			}
			int nPoint = ((int)near * 2) * ((int)near * 2);
			r /= nPoint;
			g /= nPoint;
			b /= nPoint;
			Vec3b rectColor(b, g, r);
			rectangle(draw, boundRect, rectColor, -1);


			//polylines(draw, approx, true, Scalar(255, 0, 0), 3);
			char colorString[20] = "N/A";
			detectColor(rectColor, colorString);
			putText(draw, colorString, center, FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255), 1);
		}
	}

	Mat tmp1, tmp2;
	hconcat(frame, draw, tmp2);
	return tmp2;
}