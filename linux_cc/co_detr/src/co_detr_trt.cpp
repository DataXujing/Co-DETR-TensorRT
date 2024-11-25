// co_detr_trt.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

# include "codetr.h"


float h_input[INPUT_SIZE * INPUT_SIZE * 3];
int h_output_0[1];   //1
float h_output_1[1 * 20 * 4];   //1
float h_output_2[1 * 20];   //1
int h_output_3[1 * 20];   //1

int main()
{
	codetr *CoDetr = new codetr;

	IExecutionContext* engine_context = CoDetr->load_engine("./model/test_1.plan");

	if (engine_context == nullptr)
	{
		std::cerr << "failed to create tensorrt execution context." << std::endl;
	}


	//cv2读图片
	cv::Mat image;
	image = cv::imread("./test_img/zidane.jpg", 1);

	CoDetr->preprocess(image, h_input);

	void* buffers[5];
	cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- input
	cudaMalloc(&buffers[1], 1 * sizeof(int)); //<- num_detections
	cudaMalloc(&buffers[2], 1 * 20 * 4 * sizeof(float)); //<- nmsed_boxes
	cudaMalloc(&buffers[3], 1 * 20 * sizeof(float)); //<- nmsed_scores
	cudaMalloc(&buffers[4], 1 * 20 * sizeof(int)); //<- nmsed_classes

	cudaMemcpy(buffers[0], h_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

	// -- do execute --------//
	engine_context->executeV2(buffers);

	cudaMemcpy(h_output_0, buffers[1], 1 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_1, buffers[2], 1 * 20 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_2, buffers[3], 1 * 20 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_3, buffers[4], 1 * 20 * sizeof(int), cudaMemcpyDeviceToHost);


	//std::vector<Bbox> pred_box;
	//for (int i = 0; i < 300; i++) {
	//	std::cout << "box: " << h_output_0[i * 5] << ", " << h_output_0[i * 5 + 1] << ", " << h_output_0[i * 5 + 2] << ", " << h_output_0[i * 5 + 3] << ", " << h_output_0[i * 5 + 4] << std::endl;
	//	

	//	if (h_output_0[i * 5 + 4] >= 0.80) {
	//		Bbox box;
	//		box.x1 = h_output_0[i * 5];
	//		box.y1 = h_output_0[i * 5 + 1];
	//		box.x2 = h_output_0[i * 5 + 2];
	//		box.y2 = h_output_0[i * 5 + 3];
	//		box.score = h_output_0[i * 5 + 4];
	//		box.classes = h_output_1[i];

	//		std::cout << box.classes << "," << box.score << std::endl;
	//		std::cout << box.x1 << "," << box.y1 << ", " << box.x2 << ", " << box.y2 << std::endl;


	//		pred_box.push_back(box);
	//	}
	//	
	//	//float max_score = 0.0;
	//	//int max_id = 0;
	//	//for (int j = 0; j < 80; j++) {
	//	//	if (max_score <= h_output_0[i * 80 + j]) {
	//	//		max_score = h_output_0[i * 80 + j];
	//	//		max_id = j;
	//	//	}
	//	//	//std::cout << h_output_0[i * 80 + j] << ", ";
	//	//}
	//	std::cout << "max_score: " << h_output_1[i]  << std::endl;
	//}

	std::cout << h_output_0 << std::endl;
	std::vector<Bbox> pred_box;
	for (int i = 0; i < h_output_0[0]; i++) {
		Bbox box;
		box.x1 = h_output_1[i * 4];
		box.y1 = h_output_1[i * 4 + 1];
		box.x2 = h_output_1[i * 4 + 2];
		box.y2 = h_output_1[i * 4 + 3];
		box.score = h_output_2[i];
		box.classes = h_output_3[i];

		std::cout << box.classes << "," << box.score << std::endl;
		std::cout << box.x1 << "," << box.y1 << ", " << box.x2 << ", " << box.y2 << std::endl;


		pred_box.push_back(box);
	}

	std::vector<Bbox> out = CoDetr->postprocess(pred_box, image.cols, image.rows);
	cv::Mat img = CoDetr->renderBoundingBox(image, out);

	cv::imwrite("final.jpg", img);

	// cv::namedWindow("Image", 1);//创建窗口
	// cv::imshow("Image", img);//显示图像

	// cv::waitKey(0); //等待按键

	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
	cudaFree(buffers[2]);
	cudaFree(buffers[3]);
	cudaFree(buffers[4]);

	delete engine_context;

}


