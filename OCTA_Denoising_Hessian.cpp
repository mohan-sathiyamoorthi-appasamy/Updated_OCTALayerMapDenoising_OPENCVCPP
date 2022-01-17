#include "Header_Hessian.h"

//int main()
//{
	//const char* ImageDir = new char[100];
	////Mat inputImage = imread("FinalImage.tif");
	//Mat denoisedImage;
	//
	//ImageDir = "C:\\Users\\Mohan\\source\\repos\\OCTA_Denoising_Hessian\\FinalImage.tif";
	////denoisedImage = OCTA_Denoising_Hessian(inputImage);
	//denoisedImage = OCTA_Denoising_Hessian(ImageDir);
	//imshow("ImageIn", denoisedImage);
	//waitKey(0);
	//return 0;
//}

Mat OCTA_Denoising_Hessian(Mat inputImage)
{

	//READ IMAGE AND CONVERT TO SINGLE CHANNEL GRAY IMAGE
	//Mat inputImage = imread(inputImagePath);
	//cvtColor(inputImage, inputImage, CV_BGR2GRAY);

	//imshow("Input Image", inputImage);
	//waitKey();

	
	//CONVERT TO FLOAT TYPE
	inputImage.convertTo(inputImage,CV_32FC1);

	//REMOVE THE ZERO VALUE AND REPLACED WITH THRESHOLD VALUE
	Mat zeroRemoved(300, 300, CV_32FC1,Scalar(0));
	
	float thr = 0.9;
	for (int i = 0; i < 300; i++)
	{
		for (int j = 0; j < 300; j++)
		{
			if (inputImage.at<float>(i, j) <= thr)
			{
				zeroRemoved.at<float>(i, j) = thr;
				//printf("%f\n", zeroRemoved.at<float>(i, j));
			}
			else
			{
				zeroRemoved.at<float>(i, j) = inputImage.at<float>(i, j);
			}
		}
	}

	//FIND MAX AND MIN VALUE OF INPUT IMAGE
	double min, max;
	minMaxLoc(zeroRemoved, &min, &max);
	

	//SUBTRACT MIN VALUE FROM INPUT IMAGE
	Mat SubMinInput(300, 300, CV_32FC1, Scalar(0));
	for (int i = 0; i < 300; i++)
	{
		for (int j = 0; j < 300; j++)
		{
			SubMinInput.at<float>(i, j) = zeroRemoved.at<float>(i, j) - min;
			
		}
	}
	
	//DIVIDE MAX VALUE ON INPUT IMAGE
	Mat DivMaxInput(300, 300, CV_32FC1, Scalar(0));
	for (int i = 0; i < 300; i++)
	{
		for (int j = 0; j < 300; j++)
		{
			DivMaxInput.at<float>(i, j) = SubMinInput.at<float>(i, j)/max;

		}
	}

	//COMPUTE ENHANCEMENT FOR GIVEN PARAMETERS
	float sigma[5] = { 0.5, 1.0, 1.5, 2.0, 2.5};
	int spacing = 1.0;
	int tovu = 1.5;
	Mat vesselness;
 
	for (int i = 0; i < 5; i++)
	{

		Mat Lambda2 = ImageEigenValues(DivMaxInput, sigma[i], spacing);

		Lambda2 = -1 * (Lambda2);


		Mat Lambda3;
		Lambda3 = Lambda2.clone();



		Mat Lambda_rho;
		Lambda_rho = Lambda3.clone();



		double maxVal, minVal;
		minMaxLoc(Lambda3, &minVal, &maxVal);

		double mulVal = tovu * maxVal;

		for (int i = 0; i < 300; i++)
		{
			for (int j = 0; j < 300; j++)
			{
				if (Lambda_rho.at<float>(i, j) > 0 & Lambda3.at<float>(i, j) <= mulVal)
				{
					Lambda_rho.at<float>(i, j) = mulVal;
				}

			}
		}

		for (int i = 0; i < 300; i++)
		{
			for (int j = 0; j < 300; j++)
			{
				if (Lambda_rho.at<float>(i, j) <= 0)
				{
					Lambda_rho.at<float>(i, j) = 0;
				}

			}
		}

		//imshow("Derivative", Lambda_rho);
		//waitKey();

		Mat Numerator_1 = (Lambda_rho - Lambda2) * 27;
		Mat Numerator_2 = Lambda2.mul(Lambda2);
		Mat Numerator = Numerator_1.mul(Numerator_2);
		//cout << Numerator << endl;
		Mat Denominator;
		pow((Lambda2 + Lambda_rho), 3, Denominator);

		Mat Response = Numerator / Denominator;

		for (int i = 0; i < 300; i++)
		{
			for (int j = 0; j < 300; j++)
			{
				if ((Lambda2.at<float>(i, j) >= Lambda_rho.at<float>(i, j) / 2) & (Lambda_rho.at<float>(i, j) > 0))
				{
					Response.at<float>(i, j) = 1;
				}

			}
		}

		for (int i = 0; i < 300; i++)
		{
			for (int j = 0; j < 300; j++)
			{
				if ((Lambda2.at<float>(i, j) <= 0) | (Lambda_rho.at<float>(i, j) <= 0))
				{
					Response.at<float>(i, j) = 0;
				}

			}
		}
		
		if (i == 0)
		{
			vesselness = Response.clone();
		}
		else
		{
			vesselness = maxResponse(vesselness, Response);
		}
		
	}
	//Sharpened Image
	// Unsharp Mask
	vesselness = vesselness.t();	
	Mat blurred;
	double sigma1 = 1, amount = 5;
	GaussianBlur(vesselness, blurred, Size(), sigma1, sigma1);
	Mat sharpened = vesselness*(1 + amount) + blurred*(-amount);

	//8 - Bit Conversion
	double min1, max1;
	minMaxLoc(vesselness, &min1, &max1);

	//Normalization
	
	vesselness = vesselness/max1;

	double min2, max2;
	minMaxLoc(vesselness, &min2, &max2);

	double divVal = 255 / max2;
	Mat ConvImage = vesselness * divVal;
	ConvImage.convertTo(ConvImage,CV_8UC1);

	//Adjust Second Parameter 0.5 to 2.0 to adjust Contrast
	ConvImage.convertTo(ConvImage, -1, 1.5, 0);
	//Adjust Third Parameter -20 to 20 to adjust Brightness
	ConvImage.convertTo(ConvImage, -1, 1, 10);

	//imshow("ImageIn", ConvImage);
	//waitKey(0);

	//imwrite("./Hessian.tif", ConvImage);
	return ConvImage;
}

Mat maxResponse(Mat vesselness1, Mat Response1)
{
	for (int i = 0; i < 300; i++)
	{
		for (int j = 0; j < 300; j++)
		{

			if (vesselness1.at<float>(i, j) > Response1.at<float>(i, j))
				vesselness1.at<float>(i, j) = vesselness1.at<float>(i, j);
			else
				vesselness1.at<float>(i, j) = Response1.at<float>(i, j);
		}
	}
	return vesselness1;
}

Mat ImageEigenValues(Mat DivMaxInput1, float sigma1, int spacing1)
{
	Mat Hxx;
	Mat Hyy;
	Mat Hxy;

	 //CALCULATE THE 2D HESSIAN
	Hessian2D(DivMaxInput1, sigma1, spacing1,Hxx,Hyy,Hxy);

	//Correct for Scaling
	float c = pow(sigma1, 2);

	Hxx = Hxx*c;
	Hyy = Hyy*c;
	Hxy = Hxy*c;
	
	//Reduce Computation by computation vesselness only where needed 
	Mat B1;
	B1 = -1*(Hxx + Hyy);


	Mat B2;
	B2 = (Hxx.mul(Hyy)- Hxy.mul(Hxy));
	
	//const char *fileName = "C:\\Users\\Mohan\\Documents\\Visual Studio 2013\\Projects\\HessianResult\\HessianResult\\Image.txt";
	//writeMatToFile(B2, fileName);
	
	//Create Ones Matrix
	//cout << "B1: " << B1.size() << ", B2: " << B2.size();
	Mat T(300, 300, CV_32FC1, Scalar(1));
	for (int i = 0; i < 300; i++)
	{
		for (int j = 0; j < 300; j++)
		{
			if ((B1.at<float>(i, j) < 0))
			{
				T.at<float>(i, j) = 0.0;	
			}
			 
		}
	}

	for (int i = 0; i < 300; i++)
	{
		for (int j = 0; j < 300; j++)
		{
			if ((B2.at<float>(i, j) == 0) & (B1.at<float>(i, j) == 0))
			{
				T.at<float>(i, j) = 0.0;
			}
			 
		}
	}

	//cout << T << endl;
	//imshow("Image",T.t());
	//waitKey(0);

	// Find Index Wherever 1 is appeared
	int vectorLen = countNonZero(T);
	
	Mat nz;
	T.convertTo(T, CV_8UC1);
	findNonZero(T, nz);

	Mat nzHxx(1, vectorLen, CV_32FC1, Scalar(0));
	for (int i = 0; i < vectorLen; i++)
	{
		nzHxx.at<float>(i) = Hxx.at<float>(nz.at<Point>(i));
		
	}
	 
	Mat nzHyy(1, vectorLen, CV_32FC1, Scalar(0));
	for (int i = 0; i < vectorLen; i++)
	{
		nzHyy.at<float>(i) = Hyy.at<float>(nz.at<Point>(i));

	}

	Mat nzHxy(1, vectorLen, CV_32FC1, Scalar(0));
	for (int i = 0; i < vectorLen; i++)
	{
		nzHxy.at<float>(i) = Hxy.at<float>(nz.at<Point>(i));

	}

	////EIGEN Value of Hessian
	Mat Lambda1i(1, vectorLen, CV_32FC1, Scalar(0));
	Mat Lambda2i(1, vectorLen, CV_32FC1, Scalar(0));
	EigenValofHessian(nzHxx, nzHyy, nzHxy, Lambda1i, Lambda2i);

	 

	Mat Lambda1(300,300,CV_32FC1,Scalar(0));
	Mat Lambda2(300,300, CV_32FC1, Scalar(0));

	Mat locations;
	findNonZero(T, locations); 
	for (int i = 0; i < vectorLen; i++)
	{
		Lambda1.at<float>(locations.at<Point>(i)) = Lambda1i.at<float>(i);
		//cout << Lambda1.at<float>(locations.at<Point>(i)) << endl;
	}
	
	
	for (int i = 0; i < vectorLen; i++)
	{
		Lambda2.at<float>(locations.at<Point>(i)) = Lambda2i.at<float>(i);
		//cout << Lambda1.at<float>(locations.at<Point>(i)) << endl;
	}
	
	//Check Infinite or NAN number
	Mat Mask1 = Mat(Lambda1 != Lambda1);
	int Mask1nZ = countNonZero(Mask1);

	Mat Mask2 = Mat(Lambda2 != Lambda2);
	int Mask2nZ = countNonZero(Mask2);

	Mat loc1;
	findNonZero(Mask1, loc1);
	for (int i = 0; i < Mask1nZ; i++)
	{
		Lambda1.at<float>(loc1.at<Point>(i)) = 0;
	}

	Mat loc2;
	findNonZero(Mask2, loc2);
	for (int i = 0; i < Mask2nZ; i++)
	{
		Lambda2.at<float>(loc2.at<Point>(i)) = 0;
	}
	 
	////Check Very SmallValue in Matrix
	Lambda1.setTo(0, abs(Lambda1) < 1e-4);
	Lambda2.setTo(0, abs(Lambda2) < 1e-4);

	//cout << Lambda2 << endl;
	
	return Lambda2.clone();
	
}


void EigenValofHessian(Mat HxxT1, Mat HyyT1, Mat HxyT1, Mat& Lambda1i, Mat& Lambda2i)
{
	Mat A;
	pow((HxxT1 - HyyT1),2,A);

	Mat B;
	pow((HxyT1), 2, B);
	B = B * 4;

	Mat temp;
    sqrt((A + B),temp);

	//Compute the eigen values
	Mat mu1;
	mu1 = 0.5 * (HxxT1 + HyyT1 + temp);

	Mat mu2;
	mu2 = 0.5 * (HxxT1 + HyyT1 - temp);
	
	//Sort Eigen value by abs value abs(lambda1) < abs(lambda2)
	mu1.copyTo(Lambda1i);
	mu2.copyTo(Lambda2i);
}


void Hessian2D(Mat DivMaxInput2, float sigma2, int spacing2, Mat& HxxReshaped2, Mat& HyyReshaped1, Mat& Hxy1Reshaped1)
{

	Mat GaussianFilterImage(300, 300, CV_32FC1, Scalar(0));
	if (sigma2 > 0)
		ImGaussian(DivMaxInput2, sigma2, spacing2, GaussianFilterImage);
	else
		GaussianFilterImage = DivMaxInput2;

	//const char *fileName = "C:\\Users\\Mohan\\Documents\\Visual Studio 2013\\Projects\\SampleDFTIDFT\\SampleDFTIDFT\\TEST1.txt";
	//writeMatToFile(GaussianFilterImage, fileName);
	
	// Derivates in X and Y direction
	Mat Dx;
	Mat Dy;

	//Gradient Operation-First order Derivative Y direction
	gradient2(GaussianFilterImage,Dy,'x');
	
	Mat DyReshaped(300, 300, CV_32FC1, Dy.data);
	DyReshaped = DyReshaped.t();

	//Gradient Operation-Second order Derivative  Y direction 
	Mat Hyy1;
	gradient2(DyReshaped, Hyy1, 'x');
	Mat HyyReshaped(300, 300, CV_32FC1, Hyy1.data);
	HyyReshaped1 = HyyReshaped.clone();

	//Gradient Operation-First order Derivative Y direction
	gradient2(GaussianFilterImage, Dx, 'y');
	Mat DxReshaped(300, 300, CV_32FC1, Dx.data);
	Mat Hxx1;

	//Gradient Operation-Second order Derivative Y direction
	gradient2(Dx, Hxx1, 'y');
	Mat HxxReshaped1(300, 300, CV_32FC1, Hxx1.data);
	HxxReshaped1 = HxxReshaped1.t();
	HxxReshaped2 =HxxReshaped1.clone();
	
	Mat Hxy1;
	// Gradient Operation - XY direction
	gradient2(Dx, Hxy1, 'x');
	Mat Hxy1Reshaped(300, 300, CV_32FC1, Hxy1.data);
	Hxy1Reshaped1 = Hxy1Reshaped.clone();
	
	//cout << HyyReshaped1 << endl;
}


//void writeMatToFile(cv::Mat m, const char* filename)
//{
//	ofstream fout(filename);
//
//	if (!fout)
//	{
//		cout << "File Not Opened" << endl;  return;
//	}
//
//	for (int i = 0; i<m.rows; i++)
//	{
//		for (int j = 0; j<m.cols; j++)
//		{
//			fout << m.at<float>(i, j) << "\t";
//		}
//		fout << endl;
//	}
//
//	fout.close();
//}

void ImGaussian(Mat DivMaxInput3, float sigma3, int spacing, Mat& GaussianFilterImage1)
{
	int siz = sigma3 * 6;
	int i = 1;
	GaussianBlur(DivMaxInput3, GaussianFilterImage1, Size(5,5),1,1);
}

void gradient2(Mat GaussianFilterImage, Mat &D,char option)
{
	switch (option)
	{
	case 'x':

		D.push_back(GaussianFilterImage.col(1) - GaussianFilterImage.col(0));

		for (int i = 1; i <= 298; i++)
		{
			D.push_back(((GaussianFilterImage.col(i + 1) - GaussianFilterImage.col(i - 1)) / 2));
		}

		D.push_back(GaussianFilterImage.col(299) - GaussianFilterImage.col(298));
		break;

	case 'y':

		D.push_back(GaussianFilterImage.row(1) - GaussianFilterImage.row(0));

		for (int i = 1; i <= 298; i++)
		{
			D.push_back(((GaussianFilterImage.row(i + 1) - GaussianFilterImage.row(i - 1)) / 2));
		}

		D.push_back(GaussianFilterImage.row(299) - GaussianFilterImage.row(298));

		break;
	}
}