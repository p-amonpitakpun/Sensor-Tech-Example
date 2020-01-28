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
double gamma = 0.8;
double bound_err = 10;
int erosion_size = 3;
int dilation_size = 1;
float minRadius = 1;

double circularity = 0.0;
double cr = 0.00003;

Scalar lowerBound;
Scalar upperBound;

Mat gammaTable(1, 256, CV_8U);

void setup_orange();
void findMinMax(Mat& src, Mat& mask, Scalar& min, Scalar& max);
Mat preprocess(Mat& src);
Mat ppDetect(Mat& src);
Mat postprocess(Mat& frame, Mat& image, Mat& detect);

int main(int argc, char** argv)
{
	// setup begin
	lowerBound = Scalar(13, 140, 190);
	upperBound = Scalar(21, 255, 255);

	/*try {
		printf("\r\n>> setup the ORANGE!\r\n");
		setup_orange();
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
		printf("\r\n>> can't find the ORANGE!\r\n");
	}*/


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

	uchar* p = gammaTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	// setup end

	for (;;) {
		// loop begin
		// get a new frame from camera
		Mat frame;
		//cap >> frame;
		bool captured = cap.read(frame);
		//flip(raw, frame, 1);

		if (captured) {

			image = preprocess(frame);

			detect = ppDetect(image);

			postImage = postprocess(frame, image, detect);

			imshow(wName, postImage);

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

void setup_orange()
{
	Mat orange = imread(samples::findFile("orange.png"), IMREAD_COLOR);
	printf(">> orange !!! \n");
	printf("|  width \t = %d \n", orange.rows);
	printf("|  height \t = %d \n", orange.cols);
	printf("|  channels \t = %d \n", orange.channels());

	Mat orange_blur;
	GaussianBlur(orange, orange_blur, Size(15, 15), 25, 0);

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
	lowerBound = Scalar(x_min[0] - 10,
		x_min[1],
		x_min[2] - 10);
	upperBound = Scalar(x_max[0] + 10,
		255,
		255);

	printf("\n\n\n");
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

Mat preprocess(Mat& src)
{
	Mat gblur;
	Mat mask;

	//GaussianBlur(src, gblur, Size(9, 9), 5, 0);
	bilateralFilter(src, gblur, 5, 5, 5);

	// gamma correction
	CV_Assert(gamma >= 0);
	Mat src_corrected = gblur.clone();
	LUT(gblur, gammaTable, src_corrected);

	Mat hsv;
	cvtColor(src_corrected, hsv, COLOR_BGR2HSV);

	Scalar m = mean(hsv);

	printf("MEAN  \t %.2f \t %.2f \t %.2f \t", m[0], m[1], m[2]);

	lowerBound = Scalar(14, m[1] + 70, m[2] +  60);
	upperBound = Scalar(20, m[1] + 189, m[2] + 150);

	return hsv;
}

Mat ppDetect(Mat& src)
{
	Mat mask;
	Mat dst;

	// color detection
	inRange(src, lowerBound, upperBound, mask);

	Mat mask_erode;
	//erode(mask, mask_erode, Mat(), Point(-1, -1), erosion_size);
	Mat element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	erode(mask, mask_erode, element);

	//// dilate
	Mat mask_dilate;
	//dilate(mask_erode, mask_dilate, Mat(), Point(-1, -1), dilation_size);
	Mat dilate_element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	dilate(mask_erode, mask_dilate, dilate_element);

	//return mask;
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
	Mat mergeCircle;
	if (contours.size() > 0) {

		// find the largest contour
		std::vector<Point> maxContour = contours.at(0);
		double maxArea = contourArea(maxContour);
		for (std::vector<std::vector<Point>>::iterator it = contours.begin(); it != contours.end(); ++it) {
			/* std::cout << *it; ... */
			std::vector<Point> contour = *it;
			double area = contourArea(contour);
			if (area >= maxArea) {
				maxContour = contour;
				maxArea = area;
			}
		}

		Moments m = moments(maxContour);
		circularity = (m.m00 * m.m00) / (m.m20 + m.m02) / (3.14 * 2);

		printf("CIRCULAR \t %.6f", circularity);

		// find the min enclosing circle
		Point2f center;
		float radius;
		minEnclosingCircle(maxContour, center, radius);

		circle(draw, center, radius, Scalar(0, 0, 255), 5);
	}

	printf("\n");

	Mat merge_bgr;
	cvtColor(detect, detect_bgr, COLOR_GRAY2BGR);
	//hconcat(frame, image, tmp1);
	hconcat(detect_bgr, draw, tmp2);
	//vconcat(tmp1, tmp2, dst);

	return tmp2;
	//return dst;
}