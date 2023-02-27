#pragma once
#include "Superpoint.h"


Superpoint::Superpoint()
{
}

Superpoint::~Superpoint()
{
}

bool Superpoint::GetOutput(float*** pSuperpointLocationArray, float*** pSuperpointDescArray)
{
	//Assume that there is only 1 output operator.
	long long outpixNum = mOutputDimArr[0][2] * mOutputDimArr[0][3];
	long long loc_channel = mOutputDimArr[0][1];
	long long desc_channel = mOutputDimArr[1][1];

	for (size_t chanIdx = 0; chanIdx < loc_channel; ++chanIdx){
		memcpy(pSuperpointLocationArray[0][chanIdx], mOutputValues[0].data() + outpixNum * chanIdx, sizeof(float) * outpixNum);
		memcpy(pSuperpointLocationArray[1][chanIdx], mOutputValues[0].data() + (outpixNum*loc_channel) + outpixNum * chanIdx, sizeof(float) * outpixNum);
	}
	for (size_t chanIdx = 0; chanIdx < desc_channel; ++chanIdx) {
		memcpy(pSuperpointDescArray[0][chanIdx], mOutputValues[1].data() + outpixNum * chanIdx, sizeof(float) * outpixNum);
		memcpy(pSuperpointDescArray[1][chanIdx], mOutputValues[1].data() + (outpixNum * desc_channel) + outpixNum * chanIdx, sizeof(float) * outpixNum);
	}
	return true;
}