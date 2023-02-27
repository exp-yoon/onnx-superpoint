#pragma once
#include "Twinnet.h"


Twinnet::Twinnet()
{
}

Twinnet::~Twinnet()
{
}

bool Twinnet::GetOutput(float** twinnetResultArray, float scoreCut, bool bIsOutputNormalized)
{
	//Assume that there is only 1 output operator.
	long long pixNum = mOutputUnitSizeArr[0];
	long long imgNum = mOutputValues[0].size() / pixNum;
	for (size_t imgIdx = 0; imgIdx < imgNum; ++imgIdx)
	{
		//for (size_t pixIdx = 0; pixIdx < pixNum; pixIdx += 2)
		//{
		//	if (!bIsOutputNormalized)
		//	{
		//		float val0 = exp(mOutputValues[0][imgIdx * pixNum + pixIdx]);
		//		float val1 = exp(mOutputValues[0][imgIdx * pixNum + pixIdx + 1]);
		//		float expSum = val0 + val1;
		//		mOutputValues[0][imgIdx * pixNum + pixIdx] = val0 / expSum;
		//		mOutputValues[0][imgIdx * pixNum + pixIdx + 1] = val1 / expSum;
		//	}
		//	if (mOutputValues[0][imgIdx * pixNum + pixIdx + 1] > scoreCut)
		//		twinnetResultArray[imgIdx][pixIdx / 2] = 1.;
		//	else
		//		twinnetResultArray[imgIdx][pixIdx / 2] = 0.;
		//}
		for (size_t pixIdx = 0; pixIdx < pixNum; pixIdx += 16)
		{
			__m256 val;
			float* sp = mOutputValues[0].data() + imgIdx * pixNum + pixIdx;
			if (!bIsOutputNormalized)
			{
				__m256 bkg = _mm256_set_ps(*(sp + 14), *(sp + 12), *(sp + 10), *(sp + 8), *(sp + 6), *(sp + 4), *(sp + 2), *sp);
				__m256 dft = _mm256_set_ps(*(sp + 15), *(sp + 13), *(sp + 11), *(sp + 9), *(sp + 7), *(sp + 5), *(sp + 3), *(sp + 1));
				__m256 bkgExp = _mm256_exp_ps(bkg);
				__m256 dftExp = _mm256_exp_ps(dft);
				__m256 expSum = _mm256_add_ps(bkgExp, dftExp);
				val = _mm256_div_ps(dftExp, expSum);
			}
			else
				val = _mm256_set_ps(*(sp + 15), *(sp + 13), *(sp + 11), *(sp + 9), *(sp + 7), *(sp + 5), *(sp + 3), *(sp + 1));

			//Threshold Problem
			__m256 thresh = _mm256_set1_ps(scoreCut);
			__m256 mask = _mm256_cmp_ps(val, thresh, _CMP_GT_OQ);
			__m256 one = _mm256_set1_ps(1.0);
			__m256 output = _mm256_and_ps(mask, one);
			_mm256_store_ps(twinnetResultArray[imgIdx] + pixIdx / 2, output);
		}
		//memcpy(twinnetResultArray[imgIdx], mOutputValues[0].data() + imgIdx * pixNum, sizeof(float) * pixNum);
	}
	return true;
}