#pragma once
#include <string.h>
#include <filesystem>
#include <immintrin.h>
#include "comdef.h"
#include "ONNXCore.h"


ONNXCore::ONNXCore()
{
	mSession = nullptr;
	mRunOptions = Ort::RunOptions{ nullptr };
	mSessionOptions = nullptr;
	mEnv = nullptr;
	mInputDimLenArr = nullptr;
	mOutputDimLenArr = nullptr;
	mInputDimArr = nullptr;
	mOutputDimArr = nullptr;
	mInputUnitSizeArr = nullptr;
	mOutputUnitSizeArr = nullptr;

	mIsModelLoaded = false;
	mIsDataLoaded = false;

	mInputType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	mOutputType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

	mbIs16bitModel = false;
}

ONNXCore::~ONNXCore()
{
}

bool ONNXCore::LoadModel(const wchar_t* modelPath, bool bTensorRT, bool bUseCache, const char* cachePath)
{
	mModelPath = modelPath;
	mSessionOptions = new Ort::SessionOptions();

	if (bTensorRT)
	{
		_bstr_t a(modelPath);
		const char* b = a;
		char* path_constchar = (char*)b;
		char modelDir[500] = "";
		char* tmpStr = NULL;
		char* ptr = strtok_s(path_constchar, "\\", &tmpStr);
		strcat_s(modelDir, ptr);
		strcat_s(modelDir, "\\");
		while (strstr(tmpStr, "\\") != NULL)
		{
			ptr = strtok_s(tmpStr, "\\", &tmpStr);
			strcat_s(modelDir, ptr);
			strcat_s(modelDir, "\\");
		}
		OrtTensorRTProviderOptions* trt_options = new OrtTensorRTProviderOptions();
		trt_options->device_id = 0;
		trt_options->trt_max_workspace_size = 22147483648;
		trt_options->trt_max_partition_iterations = 1000;
		trt_options->trt_min_subgraph_size = 5;
		trt_options->trt_fp16_enable = 1;
		trt_options->trt_int8_enable = 0;
		trt_options->trt_int8_use_native_calibration_table = 0;
		trt_options->trt_engine_cache_enable = (int)bUseCache;
		if (cachePath == nullptr)
		{
			trt_options->trt_engine_cache_path = (const char*)modelDir;
		}
		else
		{
			trt_options->trt_engine_cache_path = cachePath;
			std::string strCachePath = (char*)cachePath;
			auto createCacheDirRes = std::filesystem::create_directories(strCachePath);
		}
		trt_options->trt_dump_subgraphs = 1;
		mSessionOptions->AppendExecutionProvider_TensorRT(*trt_options);
		mSessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
		//mSessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
	}
	else
	{
		OrtCUDAProviderOptions* cuda_options = new OrtCUDAProviderOptions();
		cuda_options->device_id = 0;
		cuda_options->arena_extend_strategy = 1;
		cuda_options->do_copy_in_default_stream = 1;
		mSessionOptions->AppendExecutionProvider_CUDA(*cuda_options);
		mSessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	}

	std::string instanceName = "ai instance";
	mEnv = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
	mSession = new Ort::Session(*mEnv, modelPath, *mSessionOptions);

	mInputOpNum = mSession->GetInputCount();
	mOutputOpNum = mSession->GetOutputCount();

	Ort::AllocatorWithDefaultOptions allocator;

	mInputDimLenArr = new size_t[mInputOpNum];
	mOutputDimLenArr = new size_t[mOutputOpNum];
	mInputDimArr = new long long* [mInputOpNum];
	mOutputDimArr = new long long* [mOutputOpNum];
	mInputUnitSizeArr = new size_t[mInputOpNum];
	mOutputUnitSizeArr = new size_t[mOutputOpNum];

	for (int i = 0; i < mOutputValues.size(); ++i) mOutputValues[i].clear();
	for (int i = 0; i < mOutputValues16bitFloat.size(); ++i) mOutputValues16bitFloat[i].clear();
	mOutputValues.clear();
	mOutputValues16bitFloat.clear();
	mInputOpNames.clear();
	mOutputOpNames.clear();

	for (size_t i = 0; i < mInputOpNum; ++i)
	{
		mInputOpNames.push_back(mSession->GetInputName(i, allocator));
		Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(i);
		auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		mInputType = inputTensorInfo.GetElementType();
		std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
		inputDims[0] = 1;
		mInputDimLenArr[i] = inputDims.size();
		mInputDimArr[i] = new long long[mInputDimLenArr[i]];
		for (size_t j = 0; j < mInputDimLenArr[i]; ++j) mInputDimArr[i][j] = static_cast<long long>(inputDims[j]);
		mInputUnitSizeArr[i] = 1;
		for (size_t j = 1; j < mInputDimLenArr[i]; ++j) mInputUnitSizeArr[i] *= (size_t)mInputDimArr[i][j];
	}

	for (size_t i = 0; i < mOutputOpNum; ++i)
	{
		mOutputOpNames.push_back(mSession->GetOutputName(i, allocator));
		Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(i);
		auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		mOutputType = outputTensorInfo.GetElementType();
		std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
		outputDims[0] = 1;
		mOutputDimLenArr[i] = outputDims.size();
		mOutputDimArr[i] = new long long[mOutputDimLenArr[i]];
		for (size_t j = 0; j < mOutputDimLenArr[i]; ++j) mOutputDimArr[i][j] = static_cast<long long>(outputDims[j]);
		mOutputUnitSizeArr[i] = 1;
		for (size_t j = 1; j < mOutputDimLenArr[i]; ++j) mOutputUnitSizeArr[i] *= (size_t)mOutputDimArr[i][j];
	}

	if (mInputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 || mOutputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
		mbIs16bitModel = true;
	else
		mbIs16bitModel = false;
	mIsModelLoaded = true;

	for (size_t i = 0; i < mOutputOpNum; ++i)
	{
		if (mbIs16bitModel)
		{
			std::vector<uint16_t> tmpOutputValues;
			mOutputValues16bitFloat.push_back(tmpOutputValues);
		}
		else
		{
			std::vector<float> tmpOutputValues;
			mOutputValues.push_back(tmpOutputValues);
		}
	}
	return true;
}

bool ONNXCore::Run(float*** inputImgArr, int imgNum, int batch, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	if (mbIs16bitModel)
	{

	}
	else
	{
		for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
		{
			int currBatch = batch;
			if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
				currBatch = imgNum % batch;

			std::vector<std::vector<float>> inputTensorValues;
			float** singleImgArr = new float* [mInputOpNum];
			for (int opIdx = 0; opIdx < mInputOpNum; ++opIdx)
			{
				inputTensorValues.push_back(std::vector<float>());
				float* singleImg = new float[mInputUnitSizeArr[opIdx]];
				for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
				{
					int currImgIdx = batchIdx * batch + imgIdx;
					if (bNormalize)
					{
						for (int pixIdx = 0; pixIdx < mInputUnitSizeArr[opIdx]; pixIdx += 8)
						{
							float* sp = inputImgArr[opIdx][currImgIdx] + pixIdx;
							__m256 pxs = _mm256_load_ps(sp);
							__m256 val = _mm256_set1_ps(255.);
							__m256 res = _mm256_div_ps(pxs, val);
							_mm256_store_ps(singleImg + pixIdx, res);
						}
					}
					else
					{
						memcpy(singleImg, inputImgArr[opIdx][currImgIdx], sizeof(float) * mInputUnitSizeArr[opIdx]);
					}
					std::vector<float> tmp(mInputUnitSizeArr[opIdx]);
					memcpy(&tmp[0], singleImg, mInputUnitSizeArr[opIdx] * sizeof(float));
					inputTensorValues[opIdx].insert(inputTensorValues[opIdx].end(), tmp.begin(), tmp.end());
					mInputDimArr[opIdx][0] = currBatch;
					mOutputDimArr[0][0] = currBatch;
				}
				delete[] singleImg;
			}

			std::vector<Ort::Value> inputTensors;
			std::vector<Ort::Value> outputTensors;

			std::vector<std::vector<float>> outputTensorValues;
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				std::vector<float> tmp(currBatch * mOutputUnitSizeArr[i]);
				outputTensorValues.push_back(tmp);
			}

			Ort::MemoryInfo memoryInfo
				= Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
			for (int i = 0; i < mInputOpNum; ++i)
			{
				inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues[i].data(),
					mInputUnitSizeArr[i] * (size_t)currBatch, mInputDimArr[i], mInputDimLenArr[i]));
			}
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues[i].data(),
					mOutputUnitSizeArr[i] * (size_t)currBatch, mOutputDimArr[i], mOutputDimLenArr[i]));
			}

			mSession->Run(mRunOptions, mInputOpNames.data(), inputTensors.data(), mInputOpNum,
				mOutputOpNames.data(), outputTensors.data(), mOutputOpNum);

			for (size_t opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputValues[opsIdx].insert(mOutputValues[opsIdx].end(), outputTensorValues[opsIdx].begin(), outputTensorValues[opsIdx].end());
		}
	}
	return true;
}

bool ONNXCore::Run(float** inputImgArr, int imgNum, int batch, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImageChannel = (int)(mInputDimArr[0][3]);
	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	if (mbIs16bitModel)
	{

	}
	else
	{
		for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
		{
			int currBatch = batch;
			if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
				currBatch = imgNum % batch;

			std::vector<float> inputTensorValues;
			float* singleImg = new float[mInputUnitSizeArr[0]];
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				int currImgIdx = batchIdx * batch + imgIdx;
				if (bNormalize)
				{
					for (int pixIdx = 0; pixIdx < mInputUnitSizeArr[0]; pixIdx += 8)
					{
						float* sp = inputImgArr[currImgIdx] + pixIdx;
						__m256 pxs = _mm256_load_ps(sp);
						__m256 val = _mm256_set1_ps(255.);
						__m256 res = _mm256_div_ps(pxs, val);
						_mm256_store_ps(singleImg + pixIdx, res);
					}
				}
				else
				{
					memcpy(singleImg, inputImgArr[currImgIdx], sizeof(float) * mInputUnitSizeArr[0]);
				}
				std::vector<float> tmp(mInputUnitSizeArr[0]);
				memcpy(&tmp[0], singleImg, mInputUnitSizeArr[0] * sizeof(float));
				inputTensorValues.insert(inputTensorValues.end(), tmp.begin(), tmp.end());
				mInputDimArr[0][0] = currBatch;
				mOutputDimArr[0][0] = currBatch;
			}
			delete[] singleImg;

			std::vector<Ort::Value> inputTensors;
			std::vector<Ort::Value> outputTensors;

			std::vector<std::vector<float>> outputTensorValues;
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				std::vector<float> tmp(currBatch * mOutputUnitSizeArr[i]);
				outputTensorValues.push_back(tmp);
			}

			Ort::MemoryInfo memoryInfo
				= Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
			for (int i = 0; i < mInputOpNum; ++i)
			{
				inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
					mInputUnitSizeArr[i] * (size_t)currBatch, mInputDimArr[i], mInputDimLenArr[i]));
			}
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues[i].data(),
					mOutputUnitSizeArr[i] * (size_t)currBatch, mOutputDimArr[i], mOutputDimLenArr[i]));
			}

			mSession->Run(mRunOptions, mInputOpNames.data(), inputTensors.data(), mInputOpNum,
				mOutputOpNames.data(), outputTensors.data(), mOutputOpNum);

			for (size_t opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputValues[opsIdx].insert(mOutputValues[opsIdx].end(), outputTensorValues[opsIdx].begin(), outputTensorValues[opsIdx].end());
		}
	}
	return true;
}

bool ONNXCore::Run(unsigned char** inputImgArr, int imgNum, int batch, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImageChannel = (int)(mInputDimArr[0][3]);
	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	if (mbIs16bitModel)
	{

	}
	else
	{
		for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
		{
			int currBatch = batch;
			if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
				currBatch = imgNum % batch;

			std::vector<float> inputTensorValues;
			float* singleImg = new float[mInputUnitSizeArr[0]];
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				int currImgIdx = batchIdx * batch + imgIdx;
				if (bNormalize)
				{
					for (int pixIdx = 0; pixIdx < mInputUnitSizeArr[0]; ++pixIdx)
						singleImg[pixIdx] = (float)(inputImgArr[currImgIdx][pixIdx]) / 255.;
				}
				else
				{
					for (int pixIdx = 0; pixIdx < mInputUnitSizeArr[0]; ++pixIdx)
						singleImg[pixIdx] = (float)(inputImgArr[currImgIdx][pixIdx]);
				}
				std::vector<float> tmp(mInputUnitSizeArr[0]);
				memcpy(&tmp[0], singleImg, mInputUnitSizeArr[0] * sizeof(float));
				inputTensorValues.insert(inputTensorValues.end(), tmp.begin(), tmp.end());
				mInputDimArr[0][0] = currBatch;
				mOutputDimArr[0][0] = currBatch;
			}
			delete[] singleImg;

			std::vector<Ort::Value> inputTensors;
			std::vector<Ort::Value> outputTensors;

			std::vector<std::vector<float>> outputTensorValues;
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				std::vector<float> tmp(currBatch * mOutputUnitSizeArr[i]);
				outputTensorValues.push_back(tmp);
			}

			Ort::MemoryInfo memoryInfo
				= Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
			for (int i = 0; i < mInputOpNum; ++i)
			{
				inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
					mInputUnitSizeArr[i] * (size_t)currBatch, mInputDimArr[i], mInputDimLenArr[i]));
			}
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues[i].data(),
					mOutputUnitSizeArr[i] * (size_t)currBatch, mOutputDimArr[i], mOutputDimLenArr[i]));
			}

			mSession->Run(mRunOptions, mInputOpNames.data(), inputTensors.data(), mInputOpNum,
				mOutputOpNames.data(), outputTensors.data(), mOutputOpNum);

			for (size_t opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputValues[opsIdx].insert(mOutputValues[opsIdx].end(), outputTensorValues[opsIdx].begin(), outputTensorValues[opsIdx].end());
		}
	}
	return true;
}

bool ONNXCore::Run(unsigned char** inputImg, CPoint imgSize, CPoint cropSize, CPoint overlapSize,
	CPoint buffPos, int batch, bool bNormalize, bool bConvertGraytoColor)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}
	if ((cropSize.x <= overlapSize.x) || (cropSize.y <= overlapSize.y))
	{
		std::cout << "Crop size must be larger than overlap size." << std::endl;
		return false;
	}

	//float* input = new float[imgSize.x * imgSize.y];
	//const float* curr = (const float*)input;
	//float* inputConverted = new float[imgSize.x * imgSize.y];
	//float* currConverted = inputConverted;
	//__m256 divider = _mm256_set1_ps(255.);
	//for (int i = 0; i < imgSize.y; i++)
	//{
	//	for (int j = 0; j < imgSize.x; j += 8)
	//	{
	//		__m256 value = _mm256_load_ps(curr);
	//		__m256 result = _mm256_div_ps(value, divider);
	//		_mm256_store_ps(currConverted, result);
	//	}
	//}

	mImageSize = imgSize;
	mCropSize = cropSize;
	mOverlapSize = overlapSize;

	int itX = (int)(mImageSize.x / (mCropSize.x - mOverlapSize.x));
	int itY = (int)(mImageSize.y / (mCropSize.y - mOverlapSize.y));
	if (mImageSize.x - (mCropSize.x - mOverlapSize.x) * (itX - 1) > mCropSize.x) ++itX;
	if (mImageSize.y - (mCropSize.y - mOverlapSize.y) * (itY - 1) > mCropSize.y) ++itY;

	int imgNum = itX * itY;
	int nImageChannel = (int)(mInputDimArr[0][3]);
	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	if (mbIs16bitModel)
	{
		for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
		{
			int currBatch = batch;
			if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
				currBatch = imgNum % batch;

			std::vector<Ort::Float16_t> inputTensorValues;
			Ort::Float16_t* singleImg = new Ort::Float16_t[mInputUnitSizeArr[0]];
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				int currImgIdx = batchIdx * batch + imgIdx;
				int currXIdx = currImgIdx % itX;
				int currYIdx = currImgIdx / itX;

				int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
				int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
				if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
				if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

				for (int y = 0; y < cropSize.y; ++y)
				{
					for (int x = 0; x < cropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							if (bNormalize)
							{
								if (bConvertGraytoColor)
								{
									singleImg[y * cropSize.x * nImageChannel + x * nImageChannel + c].value
										= Eigen::half(float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x]) / 255.0).x;
								}
								else
								{
									singleImg[y * cropSize.x * nImageChannel + x * nImageChannel + c].value
										= Eigen::half(float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]) / 255.0).x;
								}
							}
							else
							{
								if (bConvertGraytoColor)
								{
									singleImg[y * cropSize.x * nImageChannel + x * nImageChannel + c].value
										= Eigen::half(float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x])).x;
								}
								else
								{
									singleImg[y * cropSize.x * nImageChannel + x * nImageChannel + c].value
										= Eigen::half(float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c])).x;
								}
							}
						}
					}
				}
				std::vector<Ort::Float16_t> tmp(mInputUnitSizeArr[0]);
				memcpy(&tmp[0], singleImg, mInputUnitSizeArr[0] * sizeof(Ort::Float16_t));
				//if (nImageChannel == 3)
				//{
				//	cv::Mat tmpImg(mCropSize.y, mCropSize.x, CV_32FC3);
				//	memcpy(tmpImg.data, singleImg, mInputUnitSizeArr[0] * sizeof(float));
				//}
				inputTensorValues.insert(inputTensorValues.end(), tmp.begin(), tmp.end());
				mInputDimArr[0][0] = currBatch;
				mOutputDimArr[0][0] = currBatch;
			}
			delete[] singleImg;

			std::vector<Ort::Value> inputTensors;
			std::vector<Ort::Value> outputTensors;

			std::vector<std::vector<Ort::Float16_t>> outputTensorValues;
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				std::vector<Ort::Float16_t> tmp(currBatch * mOutputUnitSizeArr[i]);
				outputTensorValues.push_back(tmp);
			}

			Ort::MemoryInfo memoryInfo
				= Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
			for (int i = 0; i < mInputOpNum; ++i)
			{
				inputTensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(memoryInfo, inputTensorValues.data(),
					mInputUnitSizeArr[i] * (size_t)currBatch, mInputDimArr[i], mInputDimLenArr[i]));
			}
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				outputTensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(memoryInfo, outputTensorValues[i].data(),
					mOutputUnitSizeArr[i] * (size_t)currBatch, mOutputDimArr[i], mOutputDimLenArr[i]));
			}

			mSession->Run(mRunOptions, mInputOpNames.data(), inputTensors.data(), mInputOpNum,
				mOutputOpNames.data(), outputTensors.data(), mOutputOpNum);

			for (size_t opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputValues16bitFloat[opsIdx].insert(mOutputValues16bitFloat[opsIdx].end(), outputTensorValues[opsIdx].begin(), outputTensorValues[opsIdx].end());
		}
	}
	else
	{
		for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
		{
			int currBatch = batch;
			if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
				currBatch = imgNum % batch;

			std::vector<float> inputTensorValues;
			float* singleImg = new float[mInputUnitSizeArr[0]];
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				int currImgIdx = batchIdx * batch + imgIdx;
				int currXIdx = currImgIdx % itX;
				int currYIdx = currImgIdx / itX;

				int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
				int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
				if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
				if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

				for (int y = 0; y < cropSize.y; ++y)
				{
					for (int x = 0; x < cropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							if (bNormalize)
							{
								if (bConvertGraytoColor)
								{
									singleImg[y * cropSize.x * nImageChannel + x * nImageChannel + c]
										= float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x]) / 255.0;
								}
								else
								{
									singleImg[y * cropSize.x * nImageChannel + x * nImageChannel + c]
										= float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]) / 255.0;
								}
							}
							else
							{
								if (bConvertGraytoColor)
								{
									singleImg[y * cropSize.x * nImageChannel + x * nImageChannel + c]
										= float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x]);
								}
								else
								{
									singleImg[y * cropSize.x * nImageChannel + x * nImageChannel + c]
										= float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]);
								}
							}
						}
					}
				}
				std::vector<float> tmp(mInputUnitSizeArr[0]);
				memcpy(&tmp[0], singleImg, mInputUnitSizeArr[0] * sizeof(float));
				//if (nImageChannel == 3)
				//{
				//	cv::Mat tmpImg(mCropSize.y, mCropSize.x, CV_32FC3);
				//	memcpy(tmpImg.data, singleImg, mInputUnitSizeArr[0] * sizeof(float));
				//}
				inputTensorValues.insert(inputTensorValues.end(), tmp.begin(), tmp.end());
				mInputDimArr[0][0] = currBatch;
				mOutputDimArr[0][0] = currBatch;
			}
			delete[] singleImg;

			std::vector<Ort::Value> inputTensors;
			std::vector<Ort::Value> outputTensors;

			std::vector<std::vector<float>> outputTensorValues;
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				std::vector<float> tmp(currBatch * mOutputUnitSizeArr[i]);
				outputTensorValues.push_back(tmp);
			}

			Ort::MemoryInfo memoryInfo
				= Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
			for (int i = 0; i < mInputOpNum; ++i)
			{
				inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
					mInputUnitSizeArr[i] * (size_t)currBatch, mInputDimArr[i], mInputDimLenArr[i]));
			}
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues[i].data(),
					mOutputUnitSizeArr[i] * (size_t)currBatch, mOutputDimArr[i], mOutputDimLenArr[i]));
			}

			mSession->Run(mRunOptions, mInputOpNames.data(), inputTensors.data(), mInputOpNum,
				mOutputOpNames.data(), outputTensors.data(), mOutputOpNum);

			for (size_t opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputValues[opsIdx].insert(mOutputValues[opsIdx].end(), outputTensorValues[opsIdx].begin(), outputTensorValues[opsIdx].end());
		}
	}

	/*segmentation result monitoring
	cv::Mat outputTest(mImageSize.y, mImageSize.x, CV_8UC1);
	for (int y = 0; y < itY; ++y)
	{
		for (int x = 0; x < itX; ++x)
		{
			for (int i = 0; i < 640 * 640; ++i)
			{
				int xOffset = (mCropSize.x - mOverlapSize.x) * x;
				int yOffset = (mCropSize.y - mOverlapSize.y) * y;
				if (xOffset + mCropSize.x > mImageSize.x) xOffset = mImageSize.x - mCropSize.x;
				if (yOffset + mCropSize.y > mImageSize.y) yOffset = mImageSize.y - mCropSize.y;
				int curY = yOffset + i / 640;
				int curX = xOffset + i % 640;

				if (mOutputValues[0][y * itX * 640 * 640 * 2 + x * 640 * 640 * 2 + 2 * i] > mOutputValues[0][y * itX * 640 * 640 * 2 + x * 640 * 640 * 2 + 2 * i + 1])
					outputTest.data[curY * mImageSize.x + curX] = 0.0f;
				else
					outputTest.data[curY * mImageSize.x + curX] = 255.0f;
			}
		}
	}
	//*/
	return true;
}

bool ONNXCore::Run(unsigned char** inputImg, CPoint imgSize, int layerNum, CPoint cropSize, CPoint overlapSize,
	CPoint buffPos, int batch, bool bNormalize, bool bConvertGraytoColor)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}
	if ((cropSize.x <= overlapSize.x) || (cropSize.y <= overlapSize.y))
	{
		std::cout << "Crop size must be larger than overlap size." << std::endl;
		return false;
	}

	mImageSize = imgSize;
	mCropSize = cropSize;
	mOverlapSize = overlapSize;

	int itX = (int)(mImageSize.x / (mCropSize.x - mOverlapSize.x));
	int itY = (int)(mImageSize.y / (mCropSize.y - mOverlapSize.y));
	if (mImageSize.x - (mCropSize.x - mOverlapSize.x) * (itX - 1) > mCropSize.x) ++itX;
	if (mImageSize.y - (mCropSize.y - mOverlapSize.y) * (itY - 1) > mCropSize.y) ++itY;

	int imgNum = itX * itY;
	int nImageChannel = (int)(mInputDimArr[0][3]);
	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	if (mbIs16bitModel)
	{
		for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
		{
			int currBatch = batch;
			if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
				currBatch = imgNum % batch;

			std::vector<Ort::Float16_t> inputTensorValues;
			Ort::Float16_t* singleImg = new Ort::Float16_t[mInputUnitSizeArr[0]];
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				int currImgIdx = batchIdx * batch + imgIdx;
				int currXIdx = currImgIdx % itX;
				int currYIdx = currImgIdx / itX;

				int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
				int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
				if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
				if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

				for (int y = 0; y < cropSize.y; ++y)
				{
					for (int x = 0; x < cropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							for (int layerIdx = 0; layerIdx < layerNum; ++layerIdx)
							{
								if (bNormalize)
								{
									if (bConvertGraytoColor)
									{
										singleImg[y * cropSize.x * nImageChannel * layerNum + x * nImageChannel * layerNum + c * layerNum + layerIdx].value
											= Eigen::half(float(inputImg[layerIdx * mImageSize.y + buffPos.y + currY + y][buffPos.x + currX + x]) / 255.0).x;
									}
									else
									{
										singleImg[y * cropSize.x * nImageChannel * layerNum + x * nImageChannel * layerNum + c * layerNum + layerIdx].value
											= Eigen::half(float(inputImg[layerIdx * mImageSize.y + buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]) / 255.0).x;
									}
								}
								else
								{
									if (bConvertGraytoColor)
									{
										singleImg[y * cropSize.x * nImageChannel * layerNum + x * nImageChannel * layerNum + c * layerNum + layerIdx].value
											= Eigen::half(float(inputImg[layerIdx * mImageSize.y + buffPos.y + currY + y][buffPos.x + currX + x])).x;
									}
									else
									{
										singleImg[y * cropSize.x * nImageChannel * layerNum + x * nImageChannel * layerNum + c * layerNum + layerIdx].value
											= Eigen::half(float(inputImg[layerIdx * mImageSize.y + buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c])).x;
									}
								}
							}
						}
					}
				}
				std::vector<Ort::Float16_t> tmp(mInputUnitSizeArr[0]);
				memcpy(&tmp[0], singleImg, mInputUnitSizeArr[0] * sizeof(Ort::Float16_t));
				//if (nImageChannel == 3)
				//{
				//	cv::Mat tmpImg(mCropSize.y, mCropSize.x, CV_32FC3);
				//	memcpy(tmpImg.data, singleImg, mInputUnitSizeArr[0] * sizeof(float));
				//}
				inputTensorValues.insert(inputTensorValues.end(), tmp.begin(), tmp.end());
				mInputDimArr[0][0] = currBatch;
				mOutputDimArr[0][0] = currBatch;
			}
			delete[] singleImg;

			std::vector<Ort::Value> inputTensors;
			std::vector<Ort::Value> outputTensors;

			std::vector<std::vector<Ort::Float16_t>> outputTensorValues;
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				std::vector<Ort::Float16_t> tmp(currBatch * mOutputUnitSizeArr[i]);
				outputTensorValues.push_back(tmp);
			}

			Ort::MemoryInfo memoryInfo
				= Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
			for (int i = 0; i < mInputOpNum; ++i)
			{
				inputTensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(memoryInfo, inputTensorValues.data(),
					mInputUnitSizeArr[i] * (size_t)currBatch, mInputDimArr[i], mInputDimLenArr[i]));
			}
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				outputTensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(memoryInfo, outputTensorValues[i].data(),
					mOutputUnitSizeArr[i] * (size_t)currBatch, mOutputDimArr[i], mOutputDimLenArr[i]));
			}

			mSession->Run(mRunOptions, mInputOpNames.data(), inputTensors.data(), mInputOpNum,
				mOutputOpNames.data(), outputTensors.data(), mOutputOpNum);

			for (size_t opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputValues16bitFloat[opsIdx].insert(mOutputValues16bitFloat[opsIdx].end(), outputTensorValues[opsIdx].begin(), outputTensorValues[opsIdx].end());
		}
	}
	else
	{
		for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
		{
			int currBatch = batch;
			if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
				currBatch = imgNum % batch;

			std::vector<float> inputTensorValues;
			float* singleImg = new float[mInputUnitSizeArr[0]];
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				int currImgIdx = batchIdx * batch + imgIdx;
				int currXIdx = currImgIdx % itX;
				int currYIdx = currImgIdx / itX;

				int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
				int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
				if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
				if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

				for (int y = 0; y < cropSize.y; ++y)
				{
					for (int x = 0; x < cropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							for (int layerIdx = 0; layerIdx < layerNum; ++layerIdx)
							{
								if (bNormalize)
								{
									if (bConvertGraytoColor)
									{
										singleImg[y * cropSize.x * nImageChannel * layerNum + x * nImageChannel * layerNum + c * layerNum + layerIdx]
											= float(inputImg[layerIdx * mImageSize.y + buffPos.y + currY + y][buffPos.x + currX + x]) / 255.0;
									}
									else
									{
										singleImg[y * cropSize.x * nImageChannel * layerNum + x * nImageChannel * layerNum + c * layerNum + layerIdx]
											= float(inputImg[layerIdx * mImageSize.y + buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]) / 255.0;
									}
								}
								else
								{
									if (bConvertGraytoColor)
									{
										singleImg[y * cropSize.x * nImageChannel * layerNum + x * nImageChannel * layerNum + c * layerNum + layerIdx]
											= float(inputImg[layerIdx * mImageSize.y + buffPos.y + currY + y][buffPos.x + currX + x]);
									}
									else
									{
										singleImg[y * cropSize.x * nImageChannel * layerNum + x * nImageChannel * layerNum + c * layerNum + layerIdx]
											= float(inputImg[layerIdx * mImageSize.y + buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]);
									}
								}
							}
						}
					}
				}
				std::vector<float> tmp(mInputUnitSizeArr[0]);
				memcpy(&tmp[0], singleImg, mInputUnitSizeArr[0] * sizeof(float));
				//if (nImageChannel == 3)
				//{
				//	cv::Mat tmpImg(mCropSize.y, mCropSize.x, CV_32FC3);
				//	memcpy(tmpImg.data, singleImg, mInputUnitSizeArr[0] * sizeof(float));
				//}
				inputTensorValues.insert(inputTensorValues.end(), tmp.begin(), tmp.end());
				mInputDimArr[0][0] = currBatch;
				mOutputDimArr[0][0] = currBatch;
			}
			delete[] singleImg;

			std::vector<Ort::Value> inputTensors;
			std::vector<Ort::Value> outputTensors;

			std::vector<std::vector<float>> outputTensorValues;
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				std::vector<float> tmp(currBatch * mOutputUnitSizeArr[i]);
				outputTensorValues.push_back(tmp);
			}

			Ort::MemoryInfo memoryInfo
				= Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
			for (int i = 0; i < mInputOpNum; ++i)
			{
				inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
					mInputUnitSizeArr[i] * (size_t)currBatch, mInputDimArr[i], mInputDimLenArr[i]));
			}
			for (int i = 0; i < mOutputOpNum; ++i)
			{
				outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues[i].data(),
					mOutputUnitSizeArr[i] * (size_t)currBatch, mOutputDimArr[i], mOutputDimLenArr[i]));
			}

			mSession->Run(mRunOptions, mInputOpNames.data(), inputTensors.data(), mInputOpNum,
				mOutputOpNames.data(), outputTensors.data(), mOutputOpNum);

			for (size_t opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputValues[opsIdx].insert(mOutputValues[opsIdx].end(), outputTensorValues[opsIdx].begin(), outputTensorValues[opsIdx].end());
		}
	}

	return true;
}

bool ONNXCore::FreeModel()
{
	for (size_t opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		delete[] mInputDimArr[opsIdx];
	for (size_t opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		delete[] mOutputDimArr[opsIdx];
	delete[] mInputDimLenArr;
	delete[] mOutputDimLenArr;
	delete[] mInputDimArr;
	delete[] mOutputDimArr;
	delete[] mInputUnitSizeArr;
	delete[] mOutputUnitSizeArr;

	if (mSession != nullptr)
		delete mSession;
	if (mSessionOptions != nullptr)
		delete mSessionOptions;
	if (mEnv != nullptr)
		delete mEnv;

	mIsModelLoaded = false;
	mIsDataLoaded = false;

	return true;
}

bool ONNXCore::IsModelLoaded()
{
	return mIsModelLoaded;
}

long long** ONNXCore::GetInputDims()
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return nullptr;
	}

	return mInputDimArr;
}

long long** ONNXCore::GetOutputDims()
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return nullptr;
	}

	return mOutputDimArr;
}

void ONNXCore::SetInputDims(long long** inputDims, size_t* inputDimLens)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return;
	}

	for (size_t opIdx = 0; opIdx < mInputOpNum; ++opIdx)
	{
		mInputDimLenArr[opIdx] = inputDimLens[opIdx];
		delete mInputDimArr[opIdx];
		mInputDimArr[opIdx] = new long long [mInputDimLenArr[opIdx]];
		mInputUnitSizeArr[opIdx] = 1;
		for (size_t lenIdx = 0; lenIdx < mInputDimLenArr[opIdx]; ++lenIdx)
		{
			mInputDimArr[opIdx][lenIdx] = inputDims[opIdx][lenIdx];
			mInputUnitSizeArr[opIdx] *= inputDims[opIdx][lenIdx];
		}	
	}
	
	return;
}

void ONNXCore::SetOutputDims(long long** outputDims, size_t* outputDimLens)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return;
	}

	for (size_t opIdx = 0; opIdx < mOutputOpNum; ++opIdx)
	{
		mOutputDimLenArr[opIdx] = outputDimLens[opIdx];
		delete mOutputDimArr[opIdx];
		mOutputDimArr[opIdx] = new long long[mOutputDimLenArr[opIdx]];
		mOutputUnitSizeArr[opIdx] = 1;
		for (size_t lenIdx = 0; lenIdx < mOutputDimLenArr[opIdx]; ++lenIdx)
		{
			mOutputDimArr[opIdx][lenIdx] = outputDims[opIdx][lenIdx];
			mOutputUnitSizeArr[opIdx] *= outputDims[opIdx][lenIdx];
		}
	}

	return;
}

bool ONNXCore::_Run()
{
	return true;
}