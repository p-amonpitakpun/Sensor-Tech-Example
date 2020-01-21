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
double bound_err = 30;
int erosion_size = 5;

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

	Mat frame;
	Mat image;
	Mat detect;
	Mat postImage;

	String wName = "videocapture";
	int timeout = 1000 / 60;

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

void setup_orange()
{
	Mat orange = imread("orange.png", IMREAD_COLOR);
	printf(">> orange !!! \n");
	printf("|  width \t = %d \n", orange.rows);
	printf("|  height \t = %d \n", orange.cols);
	printf("|  channels \t = %d \n", orange.channels());

	Mat orange_blur;
	GaussianBlur(orange, orange_blur, Size(15, 15), 75, 0);

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

	return gblur;
}

Mat ppDetect(Mat& src)
{
	Mat hsv;
	Mat mask;
	Mat mask_erode;
	Mat dst;

	// color detection
	cvtColor(src, hsv, COLOR_BGR2HSV);
	inRange(hsv, lowerBound, upperBound, mask);

	// erosion
	Mat element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	erode(mask, mask_erode, element);

	return mask_erode;
}

Mat postprocess(Mat& frame, Mat& image, Mat& detect)
{
	Mat tmp;
	Mat detect_bgr;
	Mat dst;

	cvtColor(detect, detect_bgr, COLOR_GRAY2BGR);
	hconcat(frame, image, tmp);
	hconcat(tmp, detect_bgr, dst);

	return dst;
}