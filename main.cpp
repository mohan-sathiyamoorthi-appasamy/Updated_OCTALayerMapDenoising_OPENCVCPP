#include "Header_OCTA_LayerMapping.h"
//Define Image Parameters

	//Superficial Capillary -> ILM to GCL
	// Deep Capillary -> GCL to INL
	// Outer Retina -> OPL to RPE
	// CNV -> OPL to BM
	// Choriocapillaris -> BM+10Micrometer to BM+30Micrometer
	// RPC -> ILM to ILM +100Micrometer

//Get SV Image & Segmented CSV Layers Directory
int main()
{
	char RegCSVLayerFolder[100] = ".\\Data_CSV";
	char SVImageFolder[100] = ".\\SVImage";
	int microns_value1 = 0;
	int microns_value2 = 0;

	vector<Mat> images;

	images = OCT_A_LayerMapping_Filter(RegCSVLayerFolder, SVImageFolder, 0, 2, microns_value1, microns_value2);
	//Mat MIP_Image = OCT_A_LayerMapping_Filter_MIP(RegCSVLayerFolder, SVImageFolder, 0, 3, 0, 0);
	//imshow("Image", MIP_Image);
	//imshow("Enface", images[0]);
	///imshow("Vector", images[1]);
	//waitKey(0);
}

vector<Mat> OCT_A_LayerMapping_Filter(char* RegCSVLayerFolder,char* SVImageFolder,int B1,int B2,int m1,int m2)
{
	printf("Start");
	char InputName[100];

	Mat im2_gray;

	double *meanValue;
	meanValue = (double*)malloc(noPosition*nAScan*sizeof(double));
	Mat data1;

	int Boundary1 = B1, Boundary2 = B2;
	
	//Read All Position Images
	for (int imgItr = 1; imgItr <= noPosition; imgItr++)
	{
		sprintf_s(InputName, "%s\\%04d.tif", SVImageFolder, imgItr);
		//printf("%s\n", InputName);
		im2_gray = imread(InputName, IMREAD_GRAYSCALE);
		//imshow("InputImage", im2_gray);
		//waitKey(0);
		if (im2_gray.empty())
		{
			cout << "File Name :---> " << InputName << "Image Read Empty! " << endl;
			exit(EXIT_SUCCESS);
		}

		char CSVFilename[100];
		sprintf_s(CSVFilename, "%s\\Data%04d.csv", RegCSVLayerFolder, imgItr);
		

		
		//printf("%s\n", CSVFilename);
		Mat ImageLayers = ExtractedImageLayers1(CSVFilename);


		// Layer Wise Enface Image
		Mat enfaceImage = LayerEnface(im2_gray, ImageLayers,Boundary1,Boundary2,m1,m2);
		data1.push_back(enfaceImage);
	}
	
	//Type & Size Conversion
	double min, max, divVal;
	cv::minMaxLoc(data1, &min, &max);
	divVal = 255 / max;
	Mat MulImage;
	MulImage = divVal * data1;
	Mat ConvImage;
	MulImage.convertTo(ConvImage, CV_8UC1);
	Mat FinalImage;

	resize(ConvImage, FinalImage, Size(300, 300));
	//imwrite("FinalImage.tif", FinalImage);
	imshow("Input Enface Image", FinalImage);
	waitKey(0);

	// FFT Filter - To Remove Line Artifacts
	Mat OriginalFloat;
	FinalImage.convertTo(OriginalFloat, CV_32FC1);

	Mat dftOfOriginal;
	//DFT
	takeDFT(OriginalFloat, dftOfOriginal);

	//FFTShift
	recenterDFT(dftOfOriginal);

	//Mask 
	Mat MaskImage;
	MaskProcess(dftOfOriginal, MaskImage);

	Mat MaskImage_1 = MaskImage.clone();

	//Show Spectrum
	//showDFT(MaskImage_1);

	//IFFTShift
	recenterDFT_Inverse(MaskImage);

	//IDFT
	Mat invertedDFT;
	invertDFT(MaskImage, invertedDFT);

	//Type & Size Conversion
	double min1, max1,divVal1;
	cv::minMaxLoc(invertedDFT, &min1, &max1);
	divVal1 = 255 / max1;
	Mat MulImage1;
	MulImage1 = divVal1*invertedDFT;
	Mat ConvImage1;
	MulImage1.convertTo(ConvImage1, CV_8UC1);
	
	//Normalization
	normalize(ConvImage1, ConvImage1,0,255, NORM_MINMAX);

	Mat Denoising = OCTA_Denoising_Hessian(ConvImage1);
	
	imshow("Denoising", Denoising);
	waitKey(0);

	vector<Mat> result;

	//Un-Sharp Mask
	// sharpen image using "unsharp mask" algorithm
	Mat blurred; double sigma = 1, threshold = 5, amount = 5;
	GaussianBlur(ConvImage1, blurred, Size(), sigma, sigma);
	Mat sharpened = ConvImage1 * (1 + amount) + blurred * (-amount);
	
	result.push_back(sharpened);
	result.push_back(Denoising);

	//imwrite("Sharpened.tif", sharpened);
	imshow("ArtifactsRemoved", sharpened);
	waitKey(0);

	return result;

	free(meanValue);
}

Mat OCT_A_LayerMapping_Filter_MIP(char* RegCSVLayerFolder, char* SVImageFolder, int B1, int B2, int m1, int m2)
{
	char InputName[100];
	int Boundary1 = B1, Boundary2 = B2;
	// Read CSV File
	//char* RegCSVLayerFolder = argv[1];
	//const char* RegCSVLayerFolder = "C:\\Users\\Mohan\\source\\repos\\OCT_A_Complete_Project\\RegImage\\Segmentation";
	//Read SVOCT Image Folder
	//char* SVImageFolder = argv[2];
	//const char* SVImageFolder = "C:\\Users\\Mohan\\source\\repos\\OCT_A_Complete_Project\\SVImage";
	Mat im2_gray;


	//Read Layer 1 (ILM) & Layer 5 (OPL) Region
	double* meanValue;
	meanValue = (double*)malloc(noPosition * nAScan * sizeof(double));
	Mat data1;

	//Read All Position Images
	for (int imgItr = 1; imgItr <= noPosition; imgItr++)
	{
		sprintf_s(InputName, "%s\\%04d.tif", SVImageFolder, imgItr);
		//printf("%s\n", InputName);
		im2_gray = imread(InputName, IMREAD_GRAYSCALE);
		//imshow("InputImage", im2_gray);
		//waitKey(0);

		if (im2_gray.empty())
		{
			cout << "File Name :---> " << InputName << "Image Read Empty! " << endl;
			exit(EXIT_SUCCESS);
		}

		char CSVFilename[100];
		sprintf_s(CSVFilename, "%s\\Data%04d.csv", RegCSVLayerFolder, imgItr);

		//printf("%s\n", CSVFilename);
		Mat ImageLayers = ExtractedImageLayers1(CSVFilename);

		//cout << Boundary1 << endl;
		//cout << Boundary2 << endl;
		// Layer Wise Enface Image
		Mat enfaceImage = LayerEnface_MIP(im2_gray, ImageLayers, Boundary1, Boundary2, m1, m2);
		//cout << enfaceImage << endl;
		data1.push_back(enfaceImage);
	}

	//Type & Size Conversion
	//double min, max, divVal;
	//cv::minMaxLoc(data1, &min, &max);
	//divVal = 255 / max;
	//Mat MulImage;
	//MulImage = divVal * data1;
	//Mat ConvImage;
	//MulImage.convertTo(ConvImage, CV_8UC1);
	Mat FinalImage;

	resize(data1, FinalImage, Size(300, 300));
	//imwrite("FinalImage.tif", FinalImage);
//	imshow("Input Enface Image", FinalImage);
//	waitKey(0);

	// FFT Filter - To Remove Line Artifacts
	Mat OriginalFloat;
	FinalImage.convertTo(OriginalFloat, CV_32FC1);

	Mat dftOfOriginal;
	//DFT
	takeDFT(OriginalFloat, dftOfOriginal);

	//FFTShift
	recenterDFT(dftOfOriginal);

	//Mask 
	Mat MaskImage;
	MaskProcess(dftOfOriginal, MaskImage);

	Mat MaskImage_1 = MaskImage.clone();

	//Show Spectrum
	//showDFT(MaskImage_1);

	//IFFTShift
	recenterDFT_Inverse(MaskImage);

	//IDFT
	Mat invertedDFT;
	invertDFT(MaskImage, invertedDFT);

	//Type & Size Conversion
	double min1, max1, divVal1;
	cv::minMaxLoc(invertedDFT, &min1, &max1);
	divVal1 = 255 / max1;
	Mat MulImage1;
	MulImage1 = divVal1 * invertedDFT;
	Mat ConvImage1;
	MulImage1.convertTo(ConvImage1, CV_8UC1);

	//Normalization
	normalize(ConvImage1, ConvImage1, 0, 255, NORM_MINMAX);
	imshow("Angio Enface Image", ConvImage1);
	waitKey(0);
	//imwrite("FinalImage.tif", ConvImage1);

	//Un-Sharp Mask
	// sharpen image using "unsharp mask" algorithm
	Mat blurred; double sigma = 0.5, threshold = 1, amount = 2;
	GaussianBlur(ConvImage1, blurred, Size(), sigma, sigma);
	Mat sharpened = ConvImage1 * (1 + amount) + blurred * (-amount);


	//imwrite("Line_ArtifactsRemoved.tif", sharpened);
	imshow("ArtifactsRemoved", sharpened);
	waitKey(0);

	return sharpened;

	free(meanValue);
}