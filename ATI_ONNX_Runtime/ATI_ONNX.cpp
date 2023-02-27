#pragma once
#include <atltypes.h>
#include "ATI_ONNX.h"
#include "Classification.h"
#include "Segmentation.h"
#include "Detection.h"
#include "Twinnet.h"
#include "Superpoint.h"


namespace ATI_ONNX
{
	AI::AI(int taskType)
	{
		mTaskType = taskType;
		mClassification = new Classification();
		mSegmentation = new Segmentation();
		mDetection = new Detection();
		mTwinnet = new Twinnet();
		mSuperpoint = new Superpoint();
		mVersion = "1.0.1";
	}

	AI::~AI()
	{

	}

	bool AI::LoadModel(const wchar_t* modelPath, bool bTensorRT, bool bUseCache, const char* cachePath)
	{
		bool res = false;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			res = mClassification->LoadModel(modelPath, bTensorRT, bUseCache, cachePath);
			break;
		case SEGMENTATION:
			res = mSegmentation->LoadModel(modelPath, bTensorRT, bUseCache, cachePath);
			break;
		case DETECTION:
			res = mDetection->LoadModel(modelPath, bTensorRT, bUseCache, cachePath);
			break;
		case TWINNET:
			res = mTwinnet->LoadModel(modelPath, bTensorRT, bUseCache, cachePath);
			break;
		case SUPERPOINT:
			res = mSuperpoint->LoadModel(modelPath, bTensorRT, bUseCache, cachePath);
			break;
		default:
			break;
		}
		return res;
	}

	bool AI::Run(float** inputImgArr, bool bNormalize)
	{
		std::cout << "This function was deprecated." << std::endl;
		return false;
		//bool res = false;
		//switch (mTaskType)
		//{
		//case CLASSIFICATION:
		//	res = mClassification->Run(inputImgArr, bNormalize);
		//	break;
		//case SEGMENTATION:
		//	res = mSegmentation->Run(inputImgArr, bNormalize);
		//	break;
		//case DETECTION:
		//	res = mDetection->Run(inputImgArr, bNormalize);
		//	break;
		//case TWINNET:
		//	res = mTwinnet->Run(inputImgArr, bNormalize);
		//	break;
		//default:
		//	break;
		//}
		//return res;
	}

	bool AI::Run(float*** inputImgArr, bool bNormalize)
	{
		std::cout << "This function was deprecated." << std::endl;
		return false;
		//bool res = false;
		//switch (mTaskType)
		//{
		//case CLASSIFICATION:
		//	res = mClassification->Run(inputImgArr, bNormalize);
		//	break;
		//case SEGMENTATION:
		//	res = mSegmentation->Run(inputImgArr, bNormalize);
		//	break;
		//case DETECTION:
		//	res = mDetection->Run(inputImgArr, bNormalize);
		//	break;
		//case TWINNET:
		//	res = mTwinnet->Run(inputImgArr, bNormalize);
		//	break;
		//default:
		//	break;
		//}
		//return res;
	}

	bool AI::Run(unsigned char** inputImgArr, bool bNormalize)
	{
		std::cout << "This function was deprecated." << std::endl;
		return false;
		//bool res = false;
		//switch (mTaskType)
		//{
		//case CLASSIFICATION:
		//	res = mClassification->Run(inputImgArr, bNormalize);
		//	break;
		//case SEGMENTATION:
		//	res = mSegmentation->Run(inputImgArr, bNormalize);
		//	break;
		//case DETECTION:
		//	res = mDetection->Run(inputImgArr, bNormalize);
		//	break;
		//case TWINNET:
		//	res = mTwinnet->Run(inputImgArr, bNormalize);
		//	break;
		//default:
		//	break;
		//}
		//return res;
	}

	bool AI::Run(unsigned char*** inputImgArr, bool bNormalize)
	{
		std::cout << "This function was deprecated." << std::endl;
		return false;
		//bool res = false;
		//switch (mTaskType)
		//{
		//case CLASSIFICATION:
		//	res = mClassification->Run(inputImgArr, bNormalize);
		//	break;
		//case SEGMENTATION:
		//	res = mSegmentation->Run(inputImgArr, bNormalize);
		//	break;
		//case DETECTION:
		//	res = mDetection->Run(inputImgArr, bNormalize);
		//	break;
		//case TWINNET:
		//	res = mTwinnet->Run(inputImgArr, bNormalize);
		//	break;
		//default:
		//	break;
		//}
		//return res;
	}

	bool AI::Run(unsigned char** inputImg, int imgSizeX, int imgSizeY, int cropSizeX, int cropSizeY, int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY, bool bNormalize, bool bConvertGrayToColor, bool bReloadEveryRun)
	{
		std::cout << "This function was deprecated." << std::endl;
		return false;
		//bool res = false;
		//switch (mTaskType)
		//{
		//case CLASSIFICATION:
		//	res = mClassification->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor, bReloadEveryRun);
		//	break;
		//case SEGMENTATION:
		//	res = mSegmentation->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor, bReloadEveryRun);
		//	break;
		//case DETECTION:
		//	res = mDetection->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor, bReloadEveryRun);
		//	break;
		//case TWINNET:
		//	res = mTwinnet->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor, bReloadEveryRun);
		//	break;
		//default:
		//	break;
		//}
		//return res;
	}

	bool AI::Run(float*** inputImgArr, int imgNum, int batch, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			res = mClassification->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case SEGMENTATION:
			res = mSegmentation->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case DETECTION:
			res = mDetection->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case TWINNET:
			res = mTwinnet->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case SUPERPOINT:
			res = mSuperpoint->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		default:
			break;
		}
		return res;
	}

	bool AI::Run(float** inputImgArr, int imgNum, int batch, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			res = mClassification->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case SEGMENTATION:
			res = mSegmentation->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case DETECTION:
			res = mDetection->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case TWINNET:
			res = mTwinnet->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case SUPERPOINT:
			res = mSuperpoint->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		default:
			break;
		}
		return res;
	}

	bool AI::Run(unsigned char*** inputImgArr, int batch, bool bNormalize)
	{
		std::cout << "This function was deprecated." << std::endl;
		return false;
		//bool res = false;
		//switch (mTaskType)
		//{
		//case CLASSIFICATION:
		//	res = mClassification->Run(inputImgArr, batch, bNormalize);
		//	break;
		//case SEGMENTATION:
		//	res = mSegmentation->Run(inputImgArr, batch, bNormalize);
		//	break;
		//case DETECTION:
		//	res = mDetection->Run(inputImgArr, batch, bNormalize);
		//	break;
		//case TWINNET:
		//	res = mTwinnet->Run(inputImgArr, batch, bNormalize);
		//	break;
		//default:
		//	break;
		//}
		//return res;
	}

	bool AI::Run(unsigned char** inputImgArr, int imgNum, int batch, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			res = mClassification->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case SEGMENTATION:
			res = mSegmentation->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case DETECTION:
			res = mDetection->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		case TWINNET:
			res = mTwinnet->Run(inputImgArr, imgNum, batch, bNormalize);
			break;
		default:
			break;
		}
		return res;
	}

	bool AI::Run(unsigned char** inputImg, int imgSizeX, int imgSizeY, int cropSizeX, int cropSizeY,
		int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY,
		int batch, bool bNormalize, bool bConvertGraytoColor)
	{
		bool res = false;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			res = mClassification->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY),
				CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGraytoColor);
			break;
		case SEGMENTATION:
			res = mSegmentation->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY),
				CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGraytoColor);
			break;
		case DETECTION:
			res = mDetection->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY),
				CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGraytoColor);
			break;
		case TWINNET:
			res = mTwinnet->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY),
				CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGraytoColor);
			break;
		default:
			break;
		}
		return res;
	}

	bool AI::Run(unsigned char** inputImg, int imgSizeX, int imgSizeY, int layerNum, int cropSizeX, int cropSizeY,
		int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY,
		int batch, bool bNormalize, bool bConvertGraytoColor)
	{
		bool res = false;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			res = mClassification->Run(inputImg, CPoint(imgSizeX, imgSizeY), layerNum, CPoint(cropSizeX, cropSizeY),
				CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGraytoColor);
			break;
		case SEGMENTATION:
			res = mSegmentation->Run(inputImg, CPoint(imgSizeX, imgSizeY), layerNum, CPoint(cropSizeX, cropSizeY),
				CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGraytoColor);
			break;
		case DETECTION:
			res = mDetection->Run(inputImg, CPoint(imgSizeX, imgSizeY), layerNum, CPoint(cropSizeX, cropSizeY),
				CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGraytoColor);
			break;
		case TWINNET:
			res = mTwinnet->Run(inputImg, CPoint(imgSizeX, imgSizeY), layerNum, CPoint(cropSizeX, cropSizeY),
				CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGraytoColor);
			break;
		default:
			break;
		}
		return res;
	}

	bool AI::FreeModel()
	{
		bool res = false;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			res = mClassification->FreeModel();
			break;
		case SEGMENTATION:
			res = mSegmentation->FreeModel();
			break;
		case DETECTION:
			res = mDetection->FreeModel();
			break;
		case TWINNET:
			res = mTwinnet->FreeModel();
			break;
		default:
			break;
		}
		return res;
	}

	bool AI::GetClassificationResults(float** classificationResultArray)
	{
		bool res = false;
		if (mTaskType != CLASSIFICATION)
			return res;
		else
		{
			res = mClassification->GetOutput(classificationResultArray);
			return res;
		}
	}

	bool AI::GetSegmentationResults(float*** SegmentationResultArray)
	{
		std::cout << "This function was deprecated." << std::endl;
		return false;
		//bool res = false;
		//if (mTaskType != SEGMENTATION)
		//	return res;
		//else
		//{
		//	res = mSegmentation->GetOutput(SegmentationResultArray);
		//	return res;
		//}
	}

	bool AI::GetTwinnetResults(float** twinnetResultArray, float scoreCut, bool bIsOutputNormalized)
	{
		bool res = false;
		if (mTaskType != TWINNET)
			return res;
		else
		{
			res = mTwinnet->GetOutput(twinnetResultArray, scoreCut, bIsOutputNormalized);
			return res;
		}
	}

	bool AI::GetSuperpointResults(float*** SuperpointLocationArray, float*** SuperpointDescArray)
	{
		bool res = false;
		if (mTaskType != SUPERPOINT)
			return res;
		else
		{
			res = mSuperpoint->GetOutput(SuperpointLocationArray, SuperpointDescArray);
			return res;
		}
	}

	bool AI::GetDetectionResultsByArray(DetectionResult** detectionResultArr, int* boxNumArr, float iouThresh, float scoreThresh)
	{
		std::cout << "This function was deprecated." << std::endl;
		return false;
		//bool res = false;
		//if (mTaskType != DETECTION)
		//	return res;
		//else
		//{
		//	res = mDetection->GetDetectionResults(detectionResultArr, boxNumArr, iouThresh, scoreThresh);
		//	return res;
		//}
	}

	bool AI::GetWholeImageDetectionResults(DetectionResult* detResArr, int& boxNum, int clsNum, float iouThresh, float scoreThresh)
	{
		bool res = false;
		if (mTaskType != DETECTION)
			return res;
		else
		{
			res = mDetection->GetWholeImageDetectionResults(detResArr, boxNum, clsNum, iouThresh, scoreThresh);
			return res;
		}
	}

	bool AI::GetWholeImageSegmentationResults(unsigned char* outputImg, float scoreCut, bool bIsOutputNormalized)
	{
		bool res = false;
		if (mTaskType != SEGMENTATION)
			return res;
		else
		{
			res = mSegmentation->GetWholeImageSegmentationResults(outputImg, scoreCut, bIsOutputNormalized);
			return res;
		}
	}

	bool AI::IsModelLoaded()
	{
		bool res = false;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			res = mClassification->IsModelLoaded();
			break;
		case SEGMENTATION:
			res = mSegmentation->IsModelLoaded();
			break;
		case DETECTION:
			res = mDetection->IsModelLoaded();
			break;
		case TWINNET:
			res = mTwinnet->IsModelLoaded();
			break;
		case SUPERPOINT:
			res = mSuperpoint->IsModelLoaded();
			break;
		default:
			break;
		}
		return res;
	}

	long long** AI::GetInputDims()
	{
		long long** dims = NULL;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			dims = mClassification->GetInputDims();
			break;
		case SEGMENTATION:
			dims = mSegmentation->GetInputDims();
			break;
		case DETECTION:
			dims = mDetection->GetInputDims();
			break;
		case TWINNET:
			dims = mTwinnet->GetInputDims();
			break;
		case SUPERPOINT:
			dims = mSuperpoint->GetInputDims();
			break;
		default:
			break;
		}
		return dims;
	}

	long long** AI::GetOutputDims()
	{
		long long** dims = NULL;
		switch (mTaskType)
		{
		case CLASSIFICATION:
			dims = mClassification->GetOutputDims();
			break;
		case SEGMENTATION:
			dims = mSegmentation->GetOutputDims();
			break;
		case DETECTION:
			dims = mDetection->GetOutputDims();
			break;
		case TWINNET:
			dims = mTwinnet->GetOutputDims();
			break;
		case SUPERPOINT:
			dims = mSuperpoint->GetOutputDims();
		default:
			break;
		}
		return dims;
	}

	void AI::SetInputDims(long long** inputDims, size_t* inputDimLens)
	{
		switch (mTaskType)
		{
		case CLASSIFICATION:
			mClassification->SetInputDims(inputDims, inputDimLens);
			break;
		case SEGMENTATION:
			mSegmentation->SetInputDims(inputDims, inputDimLens);
			break;
		case DETECTION:
			mDetection->SetInputDims(inputDims, inputDimLens);
			break;
		case TWINNET:
			mTwinnet->SetInputDims(inputDims, inputDimLens);
			break;
		case SUPERPOINT:
			mSuperpoint->SetInputDims(inputDims, inputDimLens);
		default:
			break;
		}
		return;
	}

	void AI::SetOutputDims(long long** outputDims, size_t* outputDimLens)
	{
		switch (mTaskType)
		{
		case CLASSIFICATION:
			mClassification->SetOutputDims(outputDims, outputDimLens);
			break;
		case SEGMENTATION:
			mSegmentation->SetOutputDims(outputDims, outputDimLens);
			break;
		case DETECTION:
			mDetection->SetOutputDims(outputDims, outputDimLens);
			break;
		case TWINNET:
			mTwinnet->SetOutputDims(outputDims, outputDimLens);
			break;
		case SUPERPOINT:
			mSuperpoint->SetOutputDims(outputDims, outputDimLens);
		default:
			break;
		}
		return;
	}

	const char* AI::GetVersion()
	{
		return mVersion;
	}
}