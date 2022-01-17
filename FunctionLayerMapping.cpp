#include "Header_OCTA_LayerMapping.h"


Mat ExtractedImageLayers1(char *RegCSVLayerFolder)
{

	ifstream inputFile(RegCSVLayerFolder);
	string current_line;

	// vector allows you to add data without knowing the exact size beforehand
	vector< vector<int> > all_data;

	// Start reading lines as long as there are lines in the file
	while (getline(inputFile, current_line)){
		// Now inside each line we need to seperate the cols
		vector<int> values;
		stringstream temp(current_line);
		string single_value;
		while (getline(temp, single_value, ',')){
			// convert the string element to a integer value
			values.push_back(atoi(single_value.c_str()));
		}
		// add the row to the complete data vector
		all_data.push_back(values);
	}


	// Now add all the data into a Mat element
	Mat vect = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_16UC1);

	// Loop over vectors and add the data
	for (int rows = 0; rows < (int)all_data.size(); rows++){
		for (int cols = 0; cols< (int)all_data[0].size(); cols++){
			vect.at<unsigned short>(rows, cols) = all_data[rows][cols];

		}
	}
	//cout << "Hi" << endl;

	return vect;
}

Mat LayerEnface(Mat im2_gray, Mat ImageLayers, int startBoundary, int endBoundary,int m1,int m2)
{
	Mat wholeInnerRetinaLayer;
	Mat data1;
	Mat data2;
	int indexItr;
	for (int aScanItr = 0; aScanItr <nAScan; aScanItr++)
	{
		double ctr = 0;
		double temp = 0;
		if (endBoundary < 5)
		{
			for (indexItr = ImageLayers.at<unsigned short>(startBoundary, aScanItr)+m1; indexItr < ImageLayers.at<unsigned short>(endBoundary, aScanItr)+m2; indexItr++)
			{
				ctr = ctr + 1;
				temp += im2_gray.at<unsigned char>(indexItr, aScanItr);

			}
		}
		else
		{
			for (indexItr = ImageLayers.at<unsigned short>(startBoundary, aScanItr); indexItr <AScanLength; indexItr++)
			{
				ctr = ctr + 1;
				temp += im2_gray.at<unsigned char>(indexItr, aScanItr);

			}
		}
		data2.push_back(temp / ctr);
	}

	transpose(data2, data2);

	return data2;
}



void writeMatToFile(cv::Mat m, const char* filename)
{
	ofstream fout(filename);

	if (!fout)
	{
		cout << "File Not Opened" << endl;  return;
	}

	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
			fout << m.at<float>(i, j) << "\t";
		}
		fout << endl;
	}

	fout.close();
}


void takeDFT(Mat& source, Mat& destination)
{
	Mat OriginalComplex[2] = { source, Mat::zeros(source.size(), CV_32F) };

	Mat dftReady;
	merge(OriginalComplex, 2, dftReady);

	Mat dftOfOriginal;
	dft(dftReady, dftOfOriginal, DFT_COMPLEX_OUTPUT);
	destination = dftOfOriginal;
}

void recenterDFT(Mat& source)
{
	int centerX = source.cols / 2;
	int centerY = source.rows / 2;

	Mat q1(source, Rect(0, 0, centerX, centerY));
	Mat q2(source, Rect(centerX, 0, centerX, centerY));
	Mat q3(source, Rect(0, centerY, centerX, centerY));
	Mat q4(source, Rect(centerX, centerY, centerX, centerY));

	Mat swapMap;

	q1.copyTo(swapMap);
	q4.copyTo(q1);
	swapMap.copyTo(q4);

	q2.copyTo(swapMap);
	q3.copyTo(q2);
	swapMap.copyTo(q3);
}
void recenterDFT_Inverse(Mat& source1)
{

	int centerX = source1.cols / 2;

	int centerY = source1.rows / 2;

	Mat q1(source1, Rect(0, 0, centerX, centerY));
	Mat q2(source1, Rect(centerX, 0, centerX, centerY));
	Mat q3(source1, Rect(0, centerY, centerX, centerY));
	Mat q4(source1, Rect(centerX, centerY, centerX, centerY));

	Mat swapMap;

	q4.copyTo(swapMap);
	q1.copyTo(q4);
	swapMap.copyTo(q1);

	q3.copyTo(swapMap);
	q2.copyTo(q3);
	swapMap.copyTo(q2);


}

void showDFT(Mat& source)
{
	Mat splitArray[2] = { Mat::zeros(source.size(), CV_32FC1), Mat::zeros(source.size(), CV_32FC1) };
	split(source, splitArray);

	Mat dftMagnitude;
	magnitude(splitArray[0], splitArray[1], dftMagnitude);

	dftMagnitude += Scalar::all(1);
	log(dftMagnitude, dftMagnitude);
	normalize(dftMagnitude, dftMagnitude, 0, 1, CV_MINMAX);

	//imshow("DFT", dftMagnitude);
	//waitKey(0);
}

void invertDFT(Mat& source, Mat& Destination)
{
	Mat inverse;
	dft(source, inverse, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	Destination = inverse;
}

void MaskProcess(Mat& source, Mat& MaskResult)
{
	Mat Mask = Mat::ones(source.size(), CV_32FC2);
	int centerMaskX = Mask.cols / 2;
	int centerMaskY = Mask.rows / 2;
	//Rectangular Mask
	for (int i = 0; i < 145; i++)
	{
		for (int j = 295; j < 310; j++)
		{
			Mask.at<float>(i, j) = 0.0;
		}
	}
	for (int i = 155; i < 300; i++)
	{
		for (int j = 295; j < 310; j++)
		{
			Mask.at<float>(i, j) = 0.0;
		}
	}
	mulSpectrums(source, Mask, MaskResult, 0);
}

Mat LayerEnface_MIP(Mat im2_gray, Mat ImageLayers, const int startBoundary, const int endBoundary, int m1, int m2)
{
	Mat wholeInnerRetinaLayer;
	Mat data1;
	Mat data2;
	int indexItr;
	for (int aScanItr = 0; aScanItr < nAScan; aScanItr++)
	{
		double ctr = 0;
		vector<unsigned char>temp;
		if (endBoundary < 7)
		{
			for (indexItr = ImageLayers.at<unsigned short>(startBoundary, aScanItr) + m1; indexItr < ImageLayers.at<unsigned short>(endBoundary, aScanItr) + m2; indexItr++)
			{
				ctr = ctr + 1;
				temp.push_back(abs(im2_gray.at<unsigned char>(indexItr, aScanItr)));

			}
		}
		else
		{
			for (indexItr = ImageLayers.at<unsigned short>(startBoundary, aScanItr); indexItr < AScanLength; indexItr++)
			{
				ctr = ctr + 1;
				temp.push_back(abs(im2_gray.at<unsigned char>(indexItr, aScanItr)));

			}
		}

		data2.push_back(*max_element(temp.begin(), temp.end()));
		temp.clear();
		//cout << data2 << endl;

	}

	transpose(data2, data2);

	return data2;
}
