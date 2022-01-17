#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

Mat ImageEigenValues(Mat DivMaxInput1, float sigma1, int spacing1);
void Hessian2D(Mat DivMaxInput2, float sigma2, int spacing2, Mat& HxxReshaped2, Mat& HyyReshaped1, Mat& Hxy1Reshaped1);
void ImGaussian(Mat DivMaxInput3, float sigma3, int spacing, Mat& GaussianFilterImage1);
void gradient2(Mat GaussianFilterImage, Mat& D, char option);
//void writeMatToFile(cv::Mat m, const char* filename);
void EigenValofHessian(Mat HxxT1, Mat HyyT1, Mat HxyT1, Mat& Lambda1i, Mat& Lambda2i);
Mat maxResponse(Mat vesselness1, Mat Response1);
Mat OCTA_Denoising_Hessian(Mat inputImage);