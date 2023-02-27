#pragma once
#include "Classification.h"


Classification::Classification()
{
}

Classification::~Classification()
{
}

bool Classification::GetOutput(float** pClassificationResultArray)
{
	//Assume that there is only 1 output operator.
	long long clsNum = mOutputDimArr[0][1];
	long long imgNum = mOutputValues[0].size() / clsNum;
	for (size_t imgIdx = 0; imgIdx < imgNum; ++imgIdx)
		memcpy(pClassificationResultArray[imgIdx], mOutputValues[0].data() + imgIdx * clsNum, sizeof(float) * clsNum);
	return true;
}