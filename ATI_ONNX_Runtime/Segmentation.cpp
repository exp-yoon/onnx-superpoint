#pragma once
#include "Segmentation.h"


Segmentation::Segmentation()
{
}

Segmentation::~Segmentation()
{
}

bool Segmentation::GetWholeImageSegmentationResults(unsigned char* outputImg, float scoreCut, bool bIsOutputNormalized)
{
	//Suppose there is only one output operation in detection tasks.
	int width = (int)mOutputDimArr[0][1];
	int height = (int)mOutputDimArr[0][2];
	int segClass = (int)mOutputDimArr[0][3];

	int itX = (int)(mImageSize.x / (mCropSize.x - mOverlapSize.x));
	int itY = (int)(mImageSize.y / (mCropSize.y - mOverlapSize.y));
	if (mImageSize.x - (mCropSize.x - mOverlapSize.x) * (itX - 1) > mCropSize.x) ++itX;
	if (mImageSize.y - (mCropSize.y - mOverlapSize.y) * (itY - 1) > mCropSize.y) ++itY;

	int imgPixelNum = width * height;

	int offsetSizeX = mCropSize.x - mOverlapSize.x;
	int offsetSizeY = mCropSize.y - mOverlapSize.y;
	int offsetLimitX = mImageSize.x - mCropSize.x;
	int offsetLimitY = mImageSize.y - mCropSize.y;

	int curOutputIdx = 0;

	if (mbIs16bitModel)
	{
		for (int y = 0; y < itY; ++y)
		{
			for (int x = 0; x < itX; ++x)
			{
				int xOffset = offsetSizeX * x;
				int yOffset = offsetSizeY * y;
				if (xOffset > offsetLimitX) xOffset = offsetLimitX;
				if (yOffset > offsetLimitY) yOffset = offsetLimitY;

				for (int i = 0; i < imgPixelNum; ++i)
				{
					float expSum = 0.0;
					int onehot = 0;
					float largestScore = std::numeric_limits<float>::lowest();
					for (int c = 0; c < segClass; ++c)
					{
						Eigen::half score_fp16;
						score_fp16.x = mOutputValues16bitFloat[0][curOutputIdx];
						float currScore = (float)Eigen::half(score_fp16);
						if (currScore > largestScore)
						{
							largestScore = currScore;
							onehot = c;
						}
						if (!bIsOutputNormalized)
							expSum += exp(currScore);
						curOutputIdx++;
					}
					if (!bIsOutputNormalized) largestScore = exp(largestScore) / expSum;
					if (largestScore < scoreCut) onehot = 0;

					int curPixelIdx = (yOffset + i / width) * mImageSize.x + xOffset + i % width;

					outputImg[curPixelIdx] = onehot;
				}
			}
		}
	}
	else
	{
		for (int y = 0; y < itY; ++y)
		{
			for (int x = 0; x < itX; ++x)
			{
				int xOffset = offsetSizeX * x;
				int yOffset = offsetSizeY * y;
				if (xOffset > offsetLimitX) xOffset = offsetLimitX;
				if (yOffset > offsetLimitY) yOffset = offsetLimitY;

				for (int i = 0; i < imgPixelNum; ++i)
				{
					float expSum = 0.0;
					int onehot = 0;
					float largestScore = std::numeric_limits<float>::lowest();
					for (int c = 0; c < segClass; ++c)
					{
						float currScore = mOutputValues[0][curOutputIdx];
						if (currScore > largestScore)
						{
							largestScore = currScore;
							onehot = c;
						}
						if (!bIsOutputNormalized)
							expSum += exp(currScore);
						curOutputIdx++;
					}
					if (!bIsOutputNormalized) largestScore = exp(largestScore) / expSum;
					if (largestScore < scoreCut) onehot = 0;

					int curPixelIdx = (yOffset + i / width) * mImageSize.x + xOffset + i % width;

					outputImg[curPixelIdx] = onehot;
				}
			}
		}
	}

	for (size_t i = 0; i < mOutputValues.size(); ++i)
		mOutputValues[i].clear();

	for (size_t i = 0; i < mOutputValues16bitFloat.size(); ++i)
		mOutputValues16bitFloat[i].clear();

	return true;
}