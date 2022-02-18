#ifndef SOUNDFIELDCAL_H
#define SOUNDFIELDCAL_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "cufft.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include "FIRFilter.h"

const int MaxSourceNum =10;
const int X_RESOLUTION=640;
const int Y_RESOLUTION=480;
const int X_PIXEL=320;
const int Y_PIXEL=240;
const int ARRAY_CHANNEL_NUM=32;
const int datalen=1024;
const int Col=32;
const int Row=1024;
const float PI=3.1415926f;
const float TwoPI=6.2831853f;
//MSVC下不支持Kernel访问const定义的float常量
__device__ const float DoublePI=6.2831853f;
__device__ const float CUDA_PI=3.1415926f;

typedef struct
{
    int SourceNums;
    int Position_X[MaxSourceNum];
    int Position_Y[MaxSourceNum];
    float SourceFreq[MaxSourceNum];
    float SPL[MaxSourceNum];
} SourceInf;
typedef struct
{
    float value;
    int Position_X;
    int Position_Y;
}MaxMinValue;

class  SoundFieldCal
{
public:
    SoundFieldCal();
    virtual ~SoundFieldCal();
    inline void   setData(float* input){this->m_oridata =input;}
    inline void   setImageThre(float threshold){this->ImageThreshold =threshold;}
    inline void   setBottomNoise(float noise){this->bottom_noise =noise;}
    inline float  getMaxfreq(){return this->maxfreqChannel->value;}
    inline float  getSPL(){return this->central_SPL;}
    inline float* getSoundResult(){return this->m_soundMatrix;}
    inline SourceInf* getSourceInf(){return  this->SourceDetect;}
    void   calculate();

private:
    void InitArray();
    void InitPlane();
    void InitHandle();
    float SPL_cal(float max_power);
    cufftComplex* Covariance_cal(float SourceFreq);
    SourceInf* PeaksFinding(float* channelData);

private:

    const float  m_sampfreq;
    const float  m_sampdist;
    const float  m_velocity;
    const int    m_datalen;
    const float  p_ref;
    float  bottom_noise;
    float  ImageThreshold;
    float  MinDetectFreq;
    float* m_oridata = nullptr;
    float* m_soundMatrix= nullptr;

    SourceInf *SourceDetect;
    FIRFilter *MyFilter;
    MaxMinValue *Max;
    MaxMinValue *maxfreqChannel;


    float  maxfreq;
    float  central_SPL;
    float  meanAmp;

    float* d_meanAmp=nullptr;
    float* h_meanAmp=nullptr;

    float* d_sound_central =nullptr;
    float* sound_central =nullptr;
    float* d_singleResized =nullptr;

    float* m_Plane_X = nullptr;
    float* m_Plane_Y = nullptr;
    float* m_Array_X = nullptr;
    float* m_Array_Y = nullptr;
    float* m_Array_Z = nullptr;
    float* d_plane_x = nullptr;
    float* d_plane_y = nullptr;
    float* d_array_x = nullptr;
    float* d_array_y = nullptr;
    float* m1_Plane_X = nullptr;
    float* m1_Plane_Y = nullptr;
    float* d1_plane_x = nullptr;
    float* d1_plane_y = nullptr;
    float* H_maxmagfreq = nullptr;

    float* d_sound_field = nullptr;
    float* d_amplitude = nullptr;
    float* d_maxmagfreq = nullptr;
    float* d_max_vector = nullptr;
    float* d_min_vector = nullptr;
    float* H_max_vector = nullptr;
    float* H_min_vector = nullptr;
    int* H_CoordXmax = nullptr;
    int* H_CoordXmin = nullptr;
    int* d_CoordXmax = nullptr;
    int* d_CoordXmin = nullptr;

    float* d_float = nullptr;
    float* channelData = nullptr;
    float* H_channelData = nullptr;

    cufftComplex* d_complex = nullptr;
    cufftComplex* d_expects = nullptr;
    cufftComplex* d_transposed = nullptr;
    cufftComplex* d_covariance = nullptr;

    cufftHandle PlanManyC2C;
    cublasHandle_t CompMatrixMultiply;

private:

    static const float ArrayPositon_X[ARRAY_CHANNEL_NUM];
    static const float ArrayPositon_Y[ARRAY_CHANNEL_NUM];
    static const float ArrayPositon_Z[ARRAY_CHANNEL_NUM];

private:
    int m = 32;//rows of matrix A ;
    int n = 32;//columns of matrix B;
    int k = 1024;//columns of matrix A and rows of matrix B;
    cufftComplex alpha;
    cufftComplex beta;

    //
    dim3 ThreadsPerBlock_0;
    dim3 BlocksPerGrid_0;

    //
    dim3 ThreadsPerBlock_1;
    dim3 BlocksPerGrid_1;

    //
    dim3 blocks;
    dim3 threads;

    //
    dim3 ThreadsPerBlock_2;
    dim3 BlocksPerGrid_2;

    //
    dim3 th;
    dim3 b;

    //used to find the maximum and minimum reduction kernel function;
    unsigned int blocks1 = Y_RESOLUTION;
    unsigned int threads1 = X_RESOLUTION;

    cudaEvent_t start_GPU, stop_GPU;
    float  time_GPU;

};


__global__ void Real2Complex(float* matrix_in,
                             cufftComplex* matrix_out);

__global__ void data_revise (cufftComplex* matrix_in,
                             cufftComplex* matrix_out);

__global__ void data_recovery(cufftComplex* matrix_in,
                              float* original,
                              cufftComplex* matrix_out);

__global__ void GetExpectedValue(cufftComplex* matrix_in,
                                 cufftComplex* matrix_out);

__global__ void Matrix_Subtraction(cufftComplex* matrix_A,
                                   cufftComplex* matrix_B,
                                   cufftComplex* matrix_C);

__global__ void Matrix_Transpose(cufftComplex* matrix_A,
                                 cufftComplex* matrix_B);

__global__ void Matrix_Conj_Transpose(cufftComplex* matrix_A,
                                      cufftComplex* matrix_B);

__global__ void Get_Cov(cuComplex* matrix_A,
                        float datasize);

__global__ void Beamforming(const float* x,
                            const float* y,
                            const float* px,
                            const float* py,
                            const float Velocity,
                            const float freq,
                            const float Radial_distance,
                            const cufftComplex* Rx,
                            float* sound_field ,
                            float* sound_central);

__global__ void Normollizing(float* matrix,
                             float max_value,
                             float min_value,
                             float threshold);

__global__ void Matrix_Max(float* matrix, float* max_value,int* position_x);
__global__ void Matrix_Min(float* matrix, float* max_value);

__device__ cuComplex Complex_Multi(cuComplex A,
                                   cuComplex B);

__device__ cuComplex Complex_Add(cuComplex A,
                                 cuComplex B);

__host__ MaxMinValue* max1(float* vector, int* index_x, int size);
__host__ MaxMinValue* min1(float* vector, int* index_x, int size);
__host__ MaxMinValue* top1Frequent(float* vector, int size);
__host__ float max(float* vector, int size);
__host__ float min(float* vector, int size);
__host__ float get_sum(float* vector ,int size);

__host__ float get_mean(float* vector ,int size);

__global__ void getSignal_amplitude(cufftComplex* fftresult,float* sign_mag);

__global__ void getPeakFreq(float* signal_mag,float freq_samp,float* freq_mag);

__global__ void RemoveEdge(float* matrix,float threshold);

__global__ void getmeanAmp(float* input,float* output);

__global__ void MatAdd(float* matrix_A,float* matrix_B);

__global__ void ChannelChoose(float* Amps,float *channelDatas,int ChannelID);

__global__ void MultiSources(float* x1,float* y1,
                             float* px,float* py,
                             float Velocity,float freq,
                             float Radial_distance,
                             cufftComplex* Rx,
                             float* sound_field,
                             float* sound_central);

__global__ void PixelResize(float* before,float* after);

__device__ float BoundCheck(float* matrix,int x,int y,int xcol,int yrow);

void quickSort(int left,int right,float data[],float index[]);
__global__ void zeros(float* input);


#endif // SOUNDFIELDCAL_H
