//Seung Hoon Lee - A01021720
//Tarea 4 - Equalization CPU version
//g++ -o equalization_CPU equalization_CPU.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

void createHistogram(cv::Mat& input, long *h_s) {
	int index;
	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			index = (int)input.at<uchar>(i,j);
			h_s[index]++;
			// h_s[0] = 1;
			// cout << h_s[index] << endl;
		}
	}
}

void normalize(cv::Mat& input, long *h_s) {
	long temp[256] = {};
	for(int i = 0; i < 256; i++) {
		temp[i] = h_s[i];
	}
	//Reinicializamos en 0 el histograma orgiinal
	for(int i = 0; i < 256; i++) {
		h_s[i] = 0;
	}
	for(int i = 0; i < 256; i++) {
		for(int j = 0; j <= i; j++) {
			h_s[i] += temp[j];
		}
		int normalizeVar = (h_s[i]*255) / (input.rows*input.cols);
        // if (normalizeVar < 0) {
        //     normalizeVar = abs(normalizeVar);
        // }
		h_s[i] = normalizeVar; 
        // cout << h_s[i] << endl; //Debug
	}
}

void equalize(cv::Mat& input, cv::Mat& output, long * h_s){
	int index;
	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			index = (int)input.at<uchar>(i,j);
			output.at<uchar>(i,j) = h_s[index];
		}
	}
}

void equalizer(cv::Mat& input, cv::Mat& output) //Le pasamos de parametro solo el input , output con CV
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
    // cout << input.rows * input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	// size_t colorBytes = input.step * input.rows;

	long h_s[256] = {};
	createHistogram(input, h_s);
	normalize(input, h_s);
	equalize(input, output, h_s);
	// cout << h_s[0] << endl; //Debug
}

int main(int argc, char *argv[])
{
	string imagePath;
	
	if(argc < 2)
		imagePath = "Images/dog1.jpeg";
  	else
  		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	cv::Mat temp(input.rows, input.cols, CV_8UC1); //Creamos una matriz temporal para cambiar el input a una griz
	cv::Mat output(input.rows, input.cols, CV_8UC1); //Se tiene que cambiar a CV_8UC1 en vez de CV_8UC3

	//Cambiamos el input a un gray
	cv::cvtColor(input, temp, CV_BGR2GRAY);

	//Call the wrapper function
	auto start_cpu =  chrono::high_resolution_clock::now();
	equalizer(temp, output);
	auto end_cpu =  chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

	printf("La cantidad de tiempo que se tarda cada ejecucion es alrededor de: %f ms\n", duration_ms.count());

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
