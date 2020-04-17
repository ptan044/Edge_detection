// EE4208 Intelligent Systems Design Assignment 2
// Canny Edge Detection Algorithm

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

using namespace cv;
using namespace std;



// Function Declarations
Mat convertBinToMat(const char* fileName, int col, int row);	// Convert .RAW binary data into a matrix
Mat reflectionPadding(Mat src, vector<vector<double> > kernel);	// Padding original image by reflection to avoid border problem after kernel convolution
double gaussian(int x, int y, double sigma); // Gaussian Formula
double createGaussianKernel(vector<vector<double> >& kernel, double sigma);	// Generates a 2D Gaussian Kernel
double gaussian(int x, int y, double sigma); // Gaussian Formula
double createLOGKernel(vector<vector<double> >& kernel, double sigma);
Mat initMat(Mat src); // Initialise dst image to be same size as src image but populated with zeros
Mat convolve2d(Mat src, vector<vector<double> > kernel, bool type); // convolution
Mat edgesIntensity(Mat fx, Mat fy); // generates edge map
Mat nonMaxSup(Mat edges);
Mat nonMaxSupDir(Mat edges, Mat edgeDir);
Mat hysteresis(Mat src, int minThresh, int maxThresh);
bool sameSign(int a, int b);
Mat getZeroCrossing(Mat lapOut);
Mat hysteresisInt(Mat src, int minThresh, int maxThresh);
Mat subPixel(Mat edges, int ratio);
Mat edgeDirection(Mat fx, Mat fy);
Mat hysteresisDirectional(Mat edges, Mat edgeDir, uchar minThresh, uchar maxThresh);

float quad_offset(uchar left, uchar middle, uchar right);


constexpr auto PI = 3.14159265359;

// Declaring dimensions of input images
int rowCana = 512, colCana = 479;
int rowFruit = 487, colFruit = 414;
int rowImg335 = 500, colImg335 = 335;
int rowLamp = 256, colLamp = 256;
int rowLeaf = 190, colLeaf = 243;



int main() {
	const char* saveDir = "C:\\Users\\Pin Da\\source\\repos\\Edge_detection\\Images\\Gennerated_images\\";
	const char* cana = "C:\\Users\\Pin Da\\source\\repos\\Canny_Edge\\Canny_Edge\\Raw Images\\cana.raw";
	const char* fruit = "C:\\Users\\Pin Da\\source\\repos\\Canny_Edge\\Canny_Edge\\Raw Images\\fruit.raw";
	const char* img335 = "C:\\Users\\Pin Da\\source\\repos\\Canny_Edge\\Canny_Edge\\Raw Images\\img335.raw";
	const char* lamp = "C:\\Users\\Pin Da\\source\\repos\\Canny_Edge\\Canny_Edge\\Raw Images\\lamp.raw";
	const char* leaf = "C:\\Users\\Pin Da\\source\\repos\\Canny_Edge\\Canny_Edge\\Raw Images\\leaf.raw";
	string edgeDetectionMethod;
	string kernelSelected = "_";
	vector<vector<double> > sobelXkernel{ {-1, 0, 1},
										{-2, 0, 2},
										{-1, 0, 1} };
	vector<vector<double> > sobelYkernel{ {1, 2, 1},
											{0, 0, 0},
											{-1, -2, -1} };

	vector<vector<double> > prewitXkernel{ {-1, 0, 1},
								{-1, 0, 1},
								{-1, 0, 1} };
	vector<vector<double> > prewitYkernel{ {1, 1, 1},
											{0, 0, 0},
											{-1, -1, -1} };
	vector<vector<double> > selectedXkernel;
	vector<vector<double> > selectedYkernel;
	

	Mat imgMat;

	bool imageValid;
	/*
	string test_image_dir = "C:\\Users\\Pin Da\\Downloads\\download.jpg";
	Mat test_image = imread(test_image_dir);
	cvtColor(test_image, test_image, COLOR_BGR2GRAY);
	cout << test_image.size();
	Mat fx = convolve2d(test_image, sobelXkernel, 1); // fx is a 32-bit signed horizontal derivative of imgSmooth
	Mat fy = convolve2d(test_image, sobelYkernel, 1); // fy is a 32-bit signed vertical derivative of imgSmooth
	//thickEdges.convertTo(thickEdges, CV_8UC1);
	//fx.convertTo(fx, CV_8UC1);
	//fy.convertTo(fy, CV_8UC1);
	//imshow("fx", fx);
	//imshow("fy", fy);
	fx = convolve2d(test_image, sobelXkernel, 1); // fx is a 32-bit signed horizontal derivative of imgSmooth
	fy = convolve2d(test_image, sobelYkernel, 1);
	Mat edgeD = edgeDirection(fx, fy);
	Mat thickEdges = edgesIntensity(fx, fy);
	//thickEdges.convertTo(thickEdges, CV_8UC1);
	//imshow("Thick Edges", thickEdges);
	//thickEdges = edgesIntensity(fx, fy);
	Mat thinEdges = nonMaxSup(thickEdges);
	thinEdges.convertTo(thinEdges, CV_8UC1);
	imshow("thin", thinEdges);

	
	Mat dirThin = nonMaxSupDir(thickEdges, edgeD);
	dirThin.convertTo(dirThin, CV_8UC1);
	imshow("dir", dirThin);
	*/
	//cout << edgeD;
	
	do {
		cout << "\nWhich image do you want to use?\n" << endl;
		cout << "1: Cana.raw \n2: Fruit.raw \n3: Img335.raw \n4: Lamp.raw \n5: Leaf.raw" << endl;
		cout << "\nPlease enter your choice: ";
		int choice;
		cin >> choice;

		switch (choice) {
		case 1:
			imgMat = convertBinToMat(cana, colCana, rowCana);
			imageValid = true;
			break;
		case 2:
			imgMat = convertBinToMat(fruit, colFruit, rowFruit);
			imageValid = true;
			break;
		case 3:
			imgMat = convertBinToMat(img335, colImg335, rowImg335);
			imageValid = true;
			break;
		case 4:
			imgMat = convertBinToMat(lamp, colLamp, rowLamp);
			imageValid = true;
			break;
		case 5:
			imgMat = convertBinToMat(leaf, colLeaf, rowLeaf);
			imageValid = true;
			break;
		default:
			cout << "\nWrong Input!\nPlease only enter an INTEGER from 1 to 5!" << endl;
			imageValid = false;
			break;
		}
		cin.clear(); // clear the cin error flags
		cin.ignore(); // clear the cin buffer
	} while (!imageValid);


	
	// Select edge detection mathod //
	bool methodValid = false;
	bool kernelValid = false;
	do {
		cout << "\nWhich edge detection method do you want to use?\n" << endl;
		cout << "1: 2nd Order \n2: Gaussian smoothinng with 1st order edge detection" << endl;
		cout << "\nPlease enter your choice: ";
		int choiceM;
		cin >> choiceM;

		switch (choiceM) {
		case 1:
			edgeDetectionMethod = "2nd Order";
			methodValid = true;
			break;
		case 2:
			edgeDetectionMethod = "1st_order";
			methodValid = true;
			
			do {
				cout << "\nWhich 1st order kernnel do you want to use?\n" << endl;
				cout << "1: Sobel \n2: Prewit" << endl;
				cout << "\nPlease enter your choice: ";
				int choiceK;
				cin >> choiceK;

				switch (choiceK) {
				case 1:
					kernelSelected = "Sobel";
					selectedXkernel = sobelXkernel;
					selectedYkernel = sobelYkernel;
					
					kernelValid = true;
					break;
				case 2:
					kernelSelected = "Prewit";
					selectedXkernel = prewitXkernel;
					selectedYkernel = prewitYkernel;
					kernelValid = true;
					break;

				default:
					cout << "\nWrong Input!\nPlease only enter an INTEGER from 1 to 2 !" << endl;
					kernelValid = false;
					break;
				}
			} while (!kernelValid);
			break;

		default:
			cout << "\nWrong Input!\nPlease only enter an INTEGER from 1 to !" << endl;
			imageValid = false;
			break;
		}
	} while (!methodValid);



	// Noise Smoothing with Gaussian Kernel //



	cout << "Enter a Standard Deviation of your choice: ";
	double sigma;
	cin >> sigma;
	imshow("Original Image", imgMat); // display original image
	//equalizeHist(imgMat, imgMat);
	imshow("Histogram Equalise", imgMat);
	vector<vector<double> > kernel; 

	if (edgeDetectionMethod == "2nd Order") {
		createLOGKernel(kernel, sigma);
		cout << "\nLOG Kernel: " << endl;
	}
	else if (edgeDetectionMethod == "1st_order") {
		createGaussianKernel(kernel, sigma);
		cout << "\nGaussian Kernel: " << endl;
	}

	// Display gaussin kernel
	for (int i = 0; i < kernel.size(); ++i) {
		for (int j = 0; j < kernel[i].size(); ++j) {
			cout << setprecision(3) << kernel[i][j] << "\t";
		}
		cout << endl;
	}


	Mat imgPadded = reflectionPadding(imgMat, kernel); // padding original image to avoid border problem
	Mat rawEdgeMap;
	Mat finalEdges;


	if (edgeDetectionMethod == "2nd Order") {
		Mat logOut = convolve2d(imgPadded, kernel, 1);
		rawEdgeMap = getZeroCrossing(logOut);
		int lowerThres = 2700;
		int upperThres = 20050;
		//string specSaveDir = saveDir + 
		finalEdges = hysteresisInt(rawEdgeMap, lowerThres, upperThres);
	}
	else if (edgeDetectionMethod == "1st_order") {
		Mat smoothedImage = convolve2d(imgPadded, kernel, 0);
		
		// 2D Sobel Kernels
		

		// 2D Prewitt Kernels



		// Convoluting smoothed image with horizontal & vertical Sobel Kernels //
		Mat fx = convolve2d(smoothedImage, selectedXkernel, 1); // fx is a 32-bit signed horizontal derivative of imgSmooth
		Mat fy = convolve2d(smoothedImage, selectedYkernel, 1); // fy is a 32-bit signed vertical derivative of imgSmooth
		Mat edgeD = edgeDirection(fx, fy);
		Mat thickEdges = edgesIntensity(fx, fy);
		Mat plotEdge;

		//Mat rawEdges = subPixel(thickEdges,2);

		Mat rawEdges = nonMaxSupDir(thickEdges,edgeD);
		Mat simpleRawEdges = nonMaxSup(thickEdges);


		int minThresh = 10; // minimum pixel value to be considered an edge
		int maxThresh = 100; // maximum pixel value to be considered an edge	
		finalEdges = hysteresis(rawEdges, minThresh, maxThresh);


		imshow("smoothed image", smoothedImage);
		thickEdges.convertTo(plotEdge, CV_8UC1);
		imshow("Thick Edges", plotEdge);
		rawEdges.convertTo(plotEdge, CV_8UC1);
		imshow("After Non_max_sup", plotEdge);
		simpleRawEdges.convertTo(plotEdge, CV_8UC1);
		imshow("After simple Non_max_sup", plotEdge);
				
	}

	imshow("Original Image", imgMat);
	//imshow("Raw Edges", rawEdgeMap);
	imshow("Final Edges", finalEdges);
	Mat highlight = Mat(imgMat.rows, imgMat.cols, CV_32SC1);
	//highlight = imgMat + finalEdges;
	//imshow("Highlight", highlight);
	waitKey(0);
	return 0;
}

// Function Definitions


// Convert .RAW binary data into a matrix
Mat convertBinToMat(const char* fileName, int col, int row) {
	ifstream input(fileName, ios::binary);
	vector<uchar> originalBuffer(istreambuf_iterator<char>(input), {});
	vector<uchar> buffer(originalBuffer.begin(), originalBuffer.end());
	Mat image = Mat(col, row, CV_8UC1); // 8 bit unsigned single cahnnel image
	memcpy(image.data, buffer.data(), buffer.size() * sizeof(unsigned char));
	//imshow("Original Image", image); // for debugging

	cout << "\n" << fileName << " has been successfully loaded!\n" << endl;

	return image;
}

// Gaussian Formula
double gaussian(int x, int y, double sigma) {
	return (1 / (2 * PI * pow(sigma, 2))) * exp(-1 * (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
}
double LOGGGG(int x, int y, double sigma) {
	float z = pow(x, 2) + pow(y, 2);
	float g = z / (2 * sigma);
	return (-1.0 / (PI * pow(sigma, 4))) * (1.0 - g) * exp(-g);

}
// Generating a 2D Gaussian Kernel to smooth the image vector

double createGaussianKernel(vector<vector<double> >& kernel, double sigma) {
	double temp = 0.0, temp2 = 0.0;
	double kernRow = 0.0, kernIndex = 0.0;
	double kernWeight = 0.0;
	if (sigma >= 1) { // checking if sigma is greater than 1
		temp = ceil(sigma); // rounding up to nearest integer
		if (fmod(temp, 2) == 0) { // checking if sigma is an even number
			temp2 = (5 * temp) + 1;
			kernRow = temp2;
		}
		if (fmod(temp, 2) == 1) { // checking if sigma is an odd number
			temp2 = 5 * temp;
			kernRow = temp2;
		}
		cout << "Standard Deviation Entered: sigma = " << sigma << endl;
		cout << "Gaussian Kernel Size: " << kernRow << " X " << kernRow << endl;
	}
	if (sigma < 1) { // checking if sigma is less than 1
		kernRow = 3; // assign smallest kernel size: 3
		cout << "Standard Deviation Entered: sigma = " << sigma << " (NON-integer: less than 1)\n" << endl;
		//cout << "Sigma = " << sigma << " is less than 1" << endl;
		cout << "Gaussian Kernel Size: " << kernRow << " X " << kernRow << endl;
	}

	kernIndex = (kernRow - 1) / 2;
	cout << "Gaussian Kernal Indexing from: " << -1 * kernIndex << " to " << kernIndex << endl;

	double filter = gaussian(-1 * kernIndex, -1 * kernIndex, sigma);
	//cout << "Gaussian: " << gaussian << endl; // for debugging

	for (int row = -1 * kernIndex; row <= kernIndex; row++) {
		vector<double> tempKern;
		for (int col = -1 * kernIndex; col <= kernIndex; col++) {
			int val = round(gaussian(row, col, sigma) / filter);
			tempKern.push_back(val);
			kernWeight += val;
		}
		kernel.push_back(tempKern);
	}


	cout << "Gaussian Kernal Weighted Sum: " << kernWeight << endl;

	return kernWeight;
}

// Padding input image by reflection so that original image dimensions are retained after kernel convolution
Mat reflectionPadding(Mat src, vector<vector<double> > kernel) {
	int border = (kernel.size() - 1) / 2;
	Mat dst(src.rows + border * 2, src.cols + border * 2, src.depth()); // constructs a larger image to fit both the image and the border
	copyMakeBorder(src, dst, border, border, border, border, BORDER_REPLICATE); // form a border in-place
	//imshow("Padded Image", padded); // for debugging

	cout << "\nReflection Padding Successful!" << endl;
	//cout << "No. of Rows of Padded Image: " << dst.rows << endl; // for debugging
	//cout << "No. of Columns of Padded Image: " << dst.cols << endl; // for debugging

	return dst;
}

// Initialises dst image to the same size as src and populated with zeros
Mat initMat(Mat src) {
	Mat dst = src.clone();
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			if (src.type() == 0) { // CV_8UC1
				dst.at<uchar>(y, x) = 0.0;
			}
			if (src.type() == 4) { // CV_32SC1
				dst.at<int>(y, x) = 0.0;
			}
		}
	}
	return dst;
}

// 2D Convolution with specified kernel
Mat convolve2d(Mat src, vector<vector<double> > kernel, bool type) {
	Mat dst; // output image
	int kernXcenter = floor(kernel.size() / 2);
	int kernYcenter = floor(kernel[0].size() / 2);

	if (type) { // checking Mat image type
		//dst = cv::Mat::zeros(src.size(), CV_32SC1); // initialise the dst image as 32-bit signed
		src.convertTo(dst, CV_32SC1);
	}
	else {
		//dst = cv::Mat::zeros(src.size(), CV_8UC1); // initialise the dst image as 8-bit unsigned
		src.convertTo(dst, CV_8UC1);
	}
	for (int row = kernXcenter; row < src.rows - kernXcenter; row++) {
		
		for (int col = kernYcenter; col < src.cols - kernYcenter; col++) {
			//cout << col << "\n";
			double total = 0;
			double weightedSum = 0;
			for (int i = -1 * kernXcenter; i <= kernXcenter; i++) {
				for (int j = -1 * kernYcenter; j <= kernYcenter; j++) {
					int kernelVal = kernel[kernXcenter + i][kernYcenter + j];
					int imgVal = int(src.at<uchar>(row + i, col + j));
					weightedSum += kernel[kernXcenter + i][kernYcenter + j];
					total += kernelVal * imgVal;

				}
			}
			if (type) {
				//cout << "Came here at least right\n";
				dst.at<int>(row, col) = total;
			}
			else
				dst.at<uchar>(row, col) = (uchar)round(total / max(weightedSum, 1.0));
		}
	}

	cout << "\nConvolution Successful!" << endl;

	return dst;
}

// Forms the edge map from x and y Sobel derivatives of an image
Mat edgesIntensity(Mat fx, Mat fy) {
	Mat dst;
	Mat _fx = abs(fx);
	Mat _fy = abs(fy);

	//fx.convertTo(fx, CV_32F);
	//fy.convertTo(fy, CV_32F);

	// magnitude = |fx| + |fy|
	dst = _fx + _fy;

	// magnitude = sqrt(fx^2 + fy^2)
	/*add(fx.mul(fx), fy.mul(fy), dst);
	sqrt(dst, dst);*/

	cout << "\nEdge Map Produced Successfully!" << endl;

	return dst; // dst is a 32-bit signed Mat
}

Mat edgeDirection(Mat fx, Mat fy) {
	Mat dst = Mat(fx.rows, fx.cols, CV_32FC1);
	for (size_t x = 0; x < fx.rows; x++) {
		for (size_t y = 0; y < fx.cols; y++) {
			int yStrength = fy.at<int>(x,y);
			int xStrength = fx.at<int>(x, y);			
			dst.at<float>(x,y) = atan2(yStrength, xStrength) * 180 / PI;	
		}
	}
	//cout << dst;
	return dst;
}

Mat nonMaxSup(Mat edges) {
	Mat horizontal, vertical, oblique1, oblique2;

	horizontal = initMat(edges);
	vertical = initMat(edges);
	oblique1 = initMat(edges);
	oblique2 = initMat(edges);
	horizontal = 0;
	vertical = 0;
	oblique1 = 0;
	oblique2 =0;
	
	for (size_t x = 1; x < edges.rows - 1; x++) {
		for (size_t y = 0; y < edges.cols; y++) {
			if ((edges.at<int>(x, y) > edges.at<int>(x - 1, y) && edges.at<int>(x, y) > edges.at<int>(x + 1, y))) {
				horizontal.at<int>(x, y) = edges.at<int>(x, y);
			}
			else
				horizontal.at<int>(x, y) = 0;
		}
	}
	cout << "here1";
	//imshow("Horizontal Edges", horizontal); // for debugging

	
	for (size_t x = 0; x < edges.rows; x++) {
		for (size_t y = 1; y < edges.cols - 1; y++) {
			if ((edges.at<int>(x, y) > edges.at<int>(x, y - 1) && edges.at<int>(x, y) > edges.at<int>(x, y + 1))) {
				vertical.at<int>(x, y) = edges.at<int>(x, y);
			}
			else
				vertical.at<int>(x, y) = 0;
		}
	}
	cout << "here2";
	//imshow("Vertical Edges", vertical); // for debugging

	
	for (size_t x = 1; x < edges.rows - 1; x++) {
		for (size_t y = 1; y < edges.cols - 1; y++) {
			if ((edges.at<int>(x, y) > edges.at<int>(x + 1, y - 1) && edges.at<int>(x, y) > edges.at<int>(x - 1, y + 1))) {
				oblique1.at<int>(x, y) = edges.at<int>(x, y);
			}
			else
				oblique1.at<int>(x, y) = 0;
		}
	}
	cout << "here3";
	//imshow("Oblique Edges (Top Left to Bottom Right)", oblique1); // for debugging

	
	for (size_t x = 1; x < edges.rows - 1; x++) {
		for (size_t y = 1; y < edges.cols - 1; y++) {
			if ((edges.at<int>(x, y) > edges.at<int>(x - 1, y - 1) && edges.at<int>(x, y) > edges.at<int>(x + 1, y + 1))) {
				oblique2.at<int>(x, y) = edges.at<int>(x, y);
			}
			else
				oblique2.at<int>(x, y) = 0;
		}
	}
	cout << "here4";
	//imshow("Oblique Edges (Bottom Left to Top Right)", oblique2); // for debugging

	Mat dst = horizontal + vertical + oblique1 + oblique2;

	cout << "\nNon-Maximum Suppression Successful!" << endl;

	return dst;
}

Mat nonMaxSupDir(Mat edges, Mat edgeDir) {

	Mat horizontal = Mat(edges.rows, edges.cols, CV_32FC1);
	Mat vertical = Mat(edges.rows, edges.cols, CV_32FC1);
	Mat oblique1 = Mat(edges.rows, edges.cols, CV_32FC1);
	Mat oblique2 = Mat(edges.rows, edges.cols, CV_32FC1);
	Mat dst = Mat(edges.rows, edges.cols, CV_32SC1);

	dst = 0;
	horizontal = 0;
	vertical = 0;
	oblique1 = 0;
	oblique2 = 0;

	for (size_t y = 0; y < edges.rows ; y++) {
		for (size_t x = 1; x < edges.cols-1; x++) {
			if ( edges.at<int>(y,x) > edges.at<int>(y, x-1) && edges.at<int>(y,x) > edges.at<int>(y,x+1) ){
				float angle = edgeDir.at<float>(y,x);
				if (angle<22.5 && angle>-22.5|| angle>157.5 ||angle<-157.5) {
					horizontal.at<int>(y, x) = 255;
					dst.at<int>(y, x) = edges.at<int>(y, x);
				}
			}
			
		}
		
	}    
	horizontal.convertTo(horizontal, CV_8UC1);

	for (size_t y = 1; y < edges.rows-1; y++) {
		for (size_t x = 0; x < edges.cols; x++) {
			if (edges.at<int>(y, x) > edges.at<int>(y-1, x ) && edges.at<int>(y, x) > edges.at<int>(y+1, x)) {
				float angle = edgeDir.at<float>(y, x);
				if (angle<112.5 && angle>67.5 || angle > -112.5 && angle<-67.5) {
					vertical.at<int>(y, x) = 255;
					dst.at<int>(y, x) = edges.at<int>(y, x);
				}
			}

		}

	}
	for (size_t y = 1; y < edges.rows - 1; y++) {
		for (size_t x = 1; x < edges.cols-1; x++) {
			if (edges.at<int>(y, x) > edges.at<int>(y+1, x -1) && edges.at<int>(y, x) > edges.at<int>(y-1, x +1)) {
				float angle = edgeDir.at<float>(y, x);
				if (angle >22.5 && angle<67.5 || angle < -112.5 && angle > -157.5) {
					oblique1.at<int>(y, x) = 255;
					dst.at<int>(y, x) = edges.at<int>(y, x);
				}
			}

		}

	}
	for (size_t y = 1; y < edges.rows - 1; y++) {
		for (size_t x = 1; x < edges.cols - 1; x++) {
			if (edges.at<int>(y, x) > edges.at<int>(y + 1, x + 1) && edges.at<int>(y, x) > edges.at<int>(y - 1, x-1)) {
				float angle = edgeDir.at<float>(y, x);
				if (angle <-22.5 && angle > -67.5 || angle > 112.5 && angle <157.5) {
					oblique2.at<int>(y, x) = 255;
					dst.at<int>(y, x) = edges.at<int>(y, x);
					
				}
			}

		}

	}





	return dst;
}

Mat subPixel(Mat edges, int ratio) {
	int maxRows = edges.rows * ratio;
	int maxCols = edges.cols * ratio;
	Mat dst = Mat(maxRows, maxCols, CV_8UC1);
	dst = 0;

	for (size_t x = 1; x < edges.rows - 1; x++) {
		for (size_t y = 0; y < edges.cols; y++) {
			int left = edges.at<uchar>(x-1, y);
			int middle = edges.at<uchar>(x , y);
			int right = edges.at<uchar>(x + 1, y);
			if (middle> left && middle > right) {
				float offset = quad_offset(left, middle, right);
				int subXPos = round((x + offset) * ratio);
			    subXPos = min(subXPos,(maxRows -1));
				int subYPos = y * ratio;

				if (subXPos >= maxRows) {
					cout << "overfloew problem\n";
					
				}
				if (subYPos >= maxCols) {
					cout << "colummnn overfloew problem\n";
				}
				dst.at<uchar>(subXPos, subYPos) = uchar(middle);
			}
		}
	}
	for (size_t x = 0; x < edges.rows; x++) {
		for (size_t y = 1; y < edges.cols - 1; y++) {
			int left = edges.at<uchar>(x, y - 1);
			int middle = edges.at<uchar>(x, y);
			int right = edges.at<uchar>(x, y + 1);
			if (middle > left && middle > right) {
				float offset = quad_offset(left, middle, right);
				int subYPos = round((y + offset) * ratio);
				subYPos = min(subYPos, (maxCols - 1));
				int subXPos = x * ratio;

				if (subXPos >= maxRows) {
					cout << "overfloew problem\n";

				}
				if (subYPos >= maxCols) {
					cout << "colummnn overfloew problem\n";
				}
				dst.at<uchar>(subXPos, subYPos) = uchar(middle);

			}
		}
	}
	return dst;
}

Mat hysteresis(Mat edges, int minThresh, int maxThresh) {
	Mat dst = Mat(edges.rows, edges.cols, CV_8UC1);

	int hystCounter = 0;
	int minGrowth = 100;
	int weakEdges = 0;
	int strongEdges = 0;
	int grownEdges = 0;
	dst = 0;
	for (int row = 0; row < edges.rows - 1; row++) {
		for (int col = 0; col < edges.cols - 1; col++) {
			// less than minimum threshold: non-edge (discarded)
			if (edges.at<int>(row, col) < minThresh) {
				edges.at<int>(row, col) = 0;
				weakEdges++;
			}
			// between min & max threshold: retained if connected to sure-edge, otherwise discarded
			/*if (src.at<uchar>(row, col) >= minThresh && src.at<uchar>(row, col) <= maxThresh) {

			}*/
			// more than maximum threshold: sure-edge (retained)
			if (edges.at<int>(row, col) > maxThresh) {
				//cout << edges.at<int>(row, col) << "\n";
				dst.at<uchar>(row, col) = 255;
				strongEdges++;
			}
		}
	}
	imshow("Befoire hysterisis", dst);
	do {
		string imgTitle = "After hysterisis applied canny_" + to_string(hystCounter);
		//imshow(imgTitle, dst);
		imwrite("C:\\Users\\Pin Da\\source\\repos\\Canny\\Canny\\Images\\" + imgTitle + ".jpg", dst);
		grownEdges = 0;		
		for (int row = 1; row < edges.rows - 1; row++) {
			for (int col = 1; col < edges.cols - 1; col++) {
				if (edges.at<int>(row, col) > maxThresh) {
					for (int xOffset = -1; xOffset < 2; xOffset++) {
						for (int yOffset = -1; yOffset < 2; yOffset++) {

							int growX = row + xOffset;
							int growY = col + yOffset;

							if (edges.at<int>(growX, growY) > minThresh && edges.at<int>(growX, growY) < maxThresh) {
								edges.at<int>(growX, growY) = maxThresh + 1;
								dst.at<uchar>(growX, growY) = 255;
								grownEdges++;
							}
							if (grownEdges > 100) {
								break;
							}
						}
					}
				}
			}
		}		
		hystCounter++;


	} while (grownEdges > minGrowth);
	return dst;
}






bool sameSign(int a, int b) {
	bool sameSign = false;
	if ((a > 0 && b > 0) || (a < 0 && b < 0)) {
		sameSign = true;
	}
	return sameSign;
}
double createLOGKernel(vector<vector<double> >& kernel, double sigma) {
	double kernRow = 0.0, kernIndex = 0.0;
	double kernWeight = 0.0;
	kernRow = 7;
	kernIndex = (kernRow - 1) / 2;
	cout << "Gaussian Kernal Indexing from: " << -1 * kernIndex << " to " << kernIndex << endl;

	double filter = LOGGGG(kernIndex, kernIndex, sigma);
	//cout << "Gaussian: " << gaussian << endl; // for debugging

	for (int row = -1 * kernIndex; row <= kernIndex; row++) {
		vector<double> tempKern;
		for (int col = -1 * kernIndex; col <= kernIndex; col++) {
			int val = (LOGGGG(row, col, sigma) / filter);
			tempKern.push_back(val);
			kernWeight += val;
		}
		kernel.push_back(tempKern);
	}
	kernel[kernIndex][kernIndex] = kernel[kernIndex][kernIndex] - kernWeight;

	cout << "Gaussian Kernal Weighted Sum: " << kernWeight << endl;

	return kernWeight;
}
Mat getZeroCrossing(Mat lapOut) {

	Mat oblique1, oblique2;
	Mat edges = Mat(lapOut.rows, lapOut.cols, CV_8UC1);
	Mat vertEdges = Mat(lapOut.rows, lapOut.cols, CV_8UC1);
	Mat horzEdges = Mat(lapOut.rows, lapOut.cols, CV_8UC1);
	Mat obliDEdges = Mat(lapOut.rows, lapOut.cols, CV_8UC1);
	Mat obliUEdges = Mat(lapOut.rows, lapOut.cols, CV_8UC1);
	Mat filteredEdges = Mat(lapOut.rows, lapOut.cols, CV_8UC1);
	Mat allEdgeStrength = Mat(lapOut.rows, lapOut.cols, CV_32SC1);
	Mat vertical = Mat(lapOut.rows, lapOut.cols, CV_32SC1);
	Mat horizontal = Mat(lapOut.rows, lapOut.cols, CV_32SC1);
	int thres = 1300;
	allEdgeStrength = 0;
	horzEdges = 0;
	vertEdges = 0;
	obliDEdges = 0;
	obliUEdges = 0;
	edges = 0;
	vertical = 0;
	horizontal = 0;
	filteredEdges = 0;
	//imshow("bob", lapOut);


	// << vertical;
	for (size_t x = 0; x < lapOut.rows; x++) {
		for (size_t y = 1; y < lapOut.cols; y++) {
			//cout << "came 1";
			int edgeA = lapOut.at<int>(x, y);
			int edgeB = lapOut.at<int>(x, y - 1);
			if (!sameSign(edgeA, edgeB)) {
				int edgeStrength = abs(edgeA - edgeB);
				horzEdges.at<char>(x, y) = 200;
				edges.at<char>(x, y) = 200;
				horizontal.at<int>(x, y) = edgeStrength;
				allEdgeStrength.at<int>(x, y) += edgeStrength;

				if (edgeStrength > thres) {
					filteredEdges.at<char>(x, y) = 200;

				}
				//edges.at<int>(x, y) = 1;
				//cout << "came 2";
			}

		}

	}
	for (size_t y = 0; y < lapOut.cols; y++) {
		for (size_t x = 1; x < lapOut.rows; x++) {
			//cout << "came 1";
			int edgeA = lapOut.at<int>(x, y);
			int edgeB = lapOut.at<int>(x - 1, y);
			if (!sameSign(edgeA, edgeB)) {
				int edgeStrength = abs(edgeA - edgeB);
				vertEdges.at<char>(x, y) = 200;
				edges.at<char>(x, y) = 200;
				vertical.at<int>(x, y) = edgeStrength;
				allEdgeStrength.at<int>(x, y) += edgeStrength;
				if (edgeStrength > thres) {
					filteredEdges.at<char>(x, y) = 200;

				}

				//edges.at<int>(x, y) = 1;
				//cout << "came 2";
			}

		}

	}
	for (size_t x = 1; x < lapOut.rows; x++) {
		for (size_t y = 1; y < lapOut.cols; y++) {
			//cout << "came 1";
			int edgeA = lapOut.at<int>(x, y);
			int edgeB = lapOut.at<int>(x - 1, y - 1);
			if (!sameSign(edgeA, edgeB)) {
				int edgeStrength = abs(edgeA - edgeB);
				obliUEdges.at<char>(x, y) = 200;
				edges.at<char>(x, y) = 200;
				allEdgeStrength.at<int>(x, y) += edgeStrength;


				if (edgeStrength > thres) {
					filteredEdges.at<char>(x, y) = 200;

				}
				//edges.at<int>(x, y) = 1;
				//cout << "came 2";
			}

		}

	}
	for (size_t x = 0; x < lapOut.rows - 1; x++) {
		for (size_t y = 1; y < lapOut.cols; y++) {
			//cout << "came 1";
			int edgeA = lapOut.at<int>(x, y);
			int edgeB = lapOut.at<int>(x + 1, y - 1);
			if (!sameSign(edgeA, edgeB)) {
				int edgeStrength = abs(edgeA - edgeB);
				obliDEdges.at<char>(x, y) = 200;
				edges.at<char>(x, y) = 200;
				allEdgeStrength.at<int>(x, y) += edgeStrength;


				if (edgeStrength > thres) {
					filteredEdges.at<char>(x, y) = 200;

				}
				//edges.at<int>(x, y) = 1;
				//cout << "came 2";
			}

		}

	}


	//cout << vertical;
	imshow("vertical edges", vertEdges);
	imshow("horizonntal edges", horzEdges);
	imshow("olqiue up edges", obliUEdges);
	imshow("oblique down edges", obliDEdges);
	imshow("edges", edges);
	imshow("filtered edges", filteredEdges);
	return allEdgeStrength;
}
Mat hysteresisInt(Mat src, int minThresh, int maxThresh) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	int hystCounter = 0;
	int minGrowth = 200;
	int weakEdges = 0;
	int strongEdges = 0;
	int grownEdges = 0;
	dst = 0;
	for (int row = 0; row < src.rows - 1; row++) {
		for (int col = 0; col < src.cols - 1; col++) {
			// less than minimum threshold: non-edge (discarded)
			if (src.at<int>(row, col) < minThresh) {
				src.at<int>(row, col) = 0;
				weakEdges++;
			}
			// between min & max threshold: retained if connected to sure-edge, otherwise discarded
			/*if (src.at<uchar>(row, col) >= minThresh && src.at<uchar>(row, col) <= maxThresh) {

			}*/
			// more than maximum threshold: sure-edge (retained)
			if (src.at<int>(row, col) > maxThresh) {
				dst.at<uchar>(row, col) = 255;
				strongEdges++;
			}
		}
	}
	do {
		string imgTitle = "After hysterisis applied_" + to_string(hystCounter);
		//imshow(imgTitle, dst);
		imwrite("C:\\Users\\Pin Da\\source\\repos\\Canny\\Canny\\Images\\" + imgTitle + ".jpg", dst);
		grownEdges = 0;
		for (int row = 0; row < src.rows - 1; row++) {
			for (int col = 0; col < src.cols - 1; col++) {
				if (src.at<int>(row, col) > maxThresh) {
					for (int xOffset = -1; xOffset < 2; xOffset++) {
						for (int yOffset = -1; yOffset < 2; yOffset++) {
							int growX = row + xOffset;
							int growY = col + yOffset;

							if (src.at<int>(growX, growY) > minThresh && src.at<int>(growX, growY) < maxThresh) {
								src.at<int>(growX, growY) = maxThresh + 1;
								dst.at<uchar>(growX, growY) = 255;
								grownEdges++;
							}
							if (grownEdges > 300) {
								break;
							}
						}
					}
				}
			}
		}
		cout << "grown edges are :" << grownEdges << "\n";
		hystCounter++;


	} while (grownEdges > minGrowth);
	return dst;
}


float quad_offset(uchar left, uchar middle, uchar right) {
	double gradOne = middle - left;
	double gradTwo = right - middle;
	double secGrad = gradTwo - gradOne;
	double offset = (-0.5 - secGrad);
	return offset;


}
Mat hysteresisDirectional(Mat edges, Mat edgeDir, uchar minThresh, uchar maxThresh) {
	Mat dst = Mat(edges.rows, edges.cols, CV_8UC1);
	cout << "direction eow" << edgeDir.rows << "\n";
	cout << "edge rows" << edges.rows << "\n";
	int hystCounter = 0;
	int minGrowth = 20;
	int weakEdges = 0;
	int strongEdges = 0;
	int grownEdges = 0;
	dst = 0;
	for (int row = 0; row < edges.rows - 1; row++) {
		for (int col = 0; col < edges.cols - 1; col++) {
			// less than minimum threshold: non-edge (discarded)
			if (edges.at<uchar>(row, col) < minThresh) {
				edges.at<uchar>(row, col) = 0;
				weakEdges++;
			}
			// between min & max threshold: retained if connected to sure-edge, otherwise discarded
			/*if (src.at<uchar>(row, col) >= minThresh && src.at<uchar>(row, col) <= maxThresh) {

			}*/
			// more than maximum threshold: sure-edge (retained)
			if (edges.at<uchar>(row, col) > maxThresh) {
				dst.at<uchar>(row, col) = 255;
				strongEdges++;
			}
		}
	}
	imshow("Hytersis: before growth", dst);
	do {
		string imgTitle = "After hysterisis applied canny_" + to_string(hystCounter);
		//imshow(imgTitle, dst);
		imwrite("C:\\Users\\Pin Da\\source\\repos\\Canny\\Canny\\Images\\" + imgTitle + ".jpg", dst);
		grownEdges = 0;
		for (int row = 1; row < edges.rows - 1; row++) {
			for (int col = 1; col < edges.cols - 1; col++) {
				if (edges.at<uchar>(row, col) > maxThresh) {
					if (edgeDir.at<float>(row, col) > 67.5) {
						for (int xOffset = -1; xOffset < 2; xOffset++) {
							int growX = row + xOffset;
							int growY = col;

							if (edges.at<uchar>(growX, growY) > minThresh && edges.at<uchar>(growX, growY) < maxThresh) {
								edges.at<uchar>(growX, growY) = maxThresh + 1;
								dst.at<uchar>(growX, growY) = 255;
								grownEdges++;
							}
							if (grownEdges > 100) {
								break;
							}

						}
					}
				}
			}
		}
		hystCounter++;


	} while (grownEdges > minGrowth);
	return dst;
}
/*
Mat hysteresisDirectional(Mat edges, Mat edgeDir, uchar minThresh, uchar maxThresh) {
	Mat dst = Mat(edges.rows, edges.cols, CV_8UC1);

	int hystCounter = 0;
	int minGrowth = 100;
	int weakEdges = 0;
	int strongEdges = 0;
	int grownEdges = 0;
	dst = 0;
	for (int row = 0; row < edges.rows - 1; row++) {
		for (int col = 0; col < edges.cols - 1; col++) {
			// less than minimum threshold: non-edge (discarded)
			if (edges.at<uchar>(row, col) < minThresh) {
				edges.at<uchar>(row, col) = 0;
				weakEdges++;
			}
			// between min & max threshold: retained if connected to sure-edge, otherwise discarded
			/*if (src.at<uchar>(row, col) >= minThresh && src.at<uchar>(row, col) <= maxThresh) {

			}
			// more than maximum threshold: sure-edge (retained)
			if (edges.at<uchar>(row, col) > maxThresh) {
				dst.at<uchar>(row, col) = 255;
				strongEdges++;
			}
		}
	}
	do {
		string imgTitle = "After hysterisis applied canny_" + to_string(hystCounter);
		//imshow(imgTitle, dst);
		imwrite("C:\\Users\\Pin Da\\source\\repos\\Canny\\Canny\\Images\\" + imgTitle + ".jpg", dst);
		grownEdges = 0;
		for (int row = 1; row < edges.rows - 2; row++) {
			for (int col = 1; col < edges.cols - 1; col++) {
				if (edges.at<uchar>(row, col) > maxThresh) {
					float angle = edgeDir.at<float>(row, col);
					cout << angle;

					string offsetType = "";
					if (angle<=22.5) {
						cout << "gowing in y\n";
						offsetType = "y";
					}
					else if (angle>= 67.5) {
						offsetType = "x";
						cout << "gowing in x\n";
						for (int xOffset = -1; xOffset < 2; xOffset++) {
							int growX = row + xOffset;
							int growY = col;
							cout << growX;
							cout << growY;
							if (edges.at<uchar>(growX, growY) > minThresh && edges.at<uchar>(growX, growY) < maxThresh) {
								edges.at<uchar>(growX, growY) = maxThresh + 1;
								dst.at<uchar>(growX, growY) = 255;
								grownEdges++;
							}

						}
					}


				}
			}
		}
		hystCounter++;


	} while (grownEdges > minGrowth);
	return dst;
}
*/