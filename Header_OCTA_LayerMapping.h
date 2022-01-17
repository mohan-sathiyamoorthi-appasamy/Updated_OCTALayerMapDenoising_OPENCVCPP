#include <iostream>
#include <conio.h>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include "Header_Hessian.h"

#define nAScan 499
#define noPosition 128
#define AScanLength 867

using namespace std;
using namespace cv;

vector<Mat> OCT_A_LayerMapping_Filter(char* RegCSVLayerFolder, char* SVImageFolder, int B1, int B2, int m1, int m2);
Mat OCT_A_LayerMapping_Filter_MIP(char* RegCSVLayerFolder, char* SVImageFolder, int B1, int B2, int m1, int m2);
//Mat FFTImageComputation(Mat img);
void writeMatToFile(Mat FinalFFTResult, const char* fileName);
Mat ExtractedImageLayers1(char *RegCSVLayerFolder);
Mat LayerEnface(Mat im2_gray, Mat ImageLayers,int Boundary1,int Boundary2,int m1,int m2);
Mat LayerEnface_MIP(Mat im2_gray, Mat ImageLayers, const int startBoundary, const int endBoundary, int m1, int m2);
void takeDFT(Mat& source, Mat& destination);
void recenterDFT(Mat& source);
void recenterDFT_Inverse(Mat& source1);
void showDFT(Mat& source);
void invertDFT(Mat& source, Mat& Destination);
void MaskProcess(Mat& source, Mat& MaskResult);
