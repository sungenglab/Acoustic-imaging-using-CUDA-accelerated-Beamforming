#ifndef FIRFILTER_H
#define FIRFILTER_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_functions.h>

/* ----------------------------------参数说明----------------------------------
 * order:       滤波阶数
 * window_type：加窗类型(1=矩形窗，2=图基窗，3=三角窗，4=汉宁窗，5=海明窗，6=布莱克曼窗)
 * fs：         采样频率
 * fp:          截止频率
 * fp1,fp2:     通带频率上下限
 * input：      输入数据指针(32通道，每通道1024个点，单通道列主序排列)
 * output：     输出数据指针(32通道，每通道1024个点，单通道列主序排列)
 * Filter_type：滤波器类型(1=低通，2=带通，3=高通)
 * ----------------------------------------------------------------------------
 */
class  FIRFilter
{
public:
    FIRFilter();
    virtual ~FIRFilter();
    void LowPassFilter (int order,int window_type,int fs,float fp,float* input,float* output);
    void HighPassFilter(int order,int window_type,int fs,float fp,float* input,float* output);
    void BandPassFilter(int order,int window_type,int fs,float fp1,float fp2,float* input,float* output);

private:

    void   FIR_Coefficient(int order, int Filter_type, int window_type, int fs, float* h, float fln, float fhn);
    float  window(int window_type, int n, int i);
    void   Filter_Cal(float* coeff,int order,float* input,float* output);

private:

    float*  m_input  = nullptr;
    float*  m_output = nullptr;
    float*  m_coeff  = nullptr;
    float*  d_coeff  = nullptr;
    float*  d_input  = nullptr;
    float*  d_output = nullptr;
    float*  d_toeplitz = nullptr;

    dim3 threads;
    dim3 blocks;

    cudaEvent_t start_GPU;
    cudaEvent_t stop_GPU;
    float  time_GPU;
    cublasHandle_t MatrixMultiply;
private:

};


__global__ void Toeplitz(float* coeff,float* coeff_Toeplitz,int order);

#endif // FIRFILTER_H
