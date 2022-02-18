#ifndef SOURCEFINDING_H
#define SOURCEFINDING_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include <helper_cuda.h>
#include <helper_functions.h>

const int MaxNum =10;
const int ARRAYNUM =32;
const int len=1024;

typedef struct
{
    int SourceNums;
    float SourceFreq[MaxNum];
} Sources;

typedef struct
{
    float PeakFreq;
    int channelID;
}Channels;

class SourceFinding
{
public:
    SourceFinding();
    ~SourceFinding();
    inline void   setData(float* input){this->m_oridata =input;}
    inline Sources* getSourceInf(){return  this->SourceDetect;}
    void finding();

private:
    Sources *PeaksFinding(float* channelData,float sampFreq);
    void quickSort(int left,int right,float data[],float index[]);
    Channels* Top1Frequent(float* vector, int size);
private:
    float* m_oridata;
    Sources* SourceDetect;
    Channels* UsedChannnel;

    cufftHandle PlanManyC2C;
    float m_sampfreq;
    float MinDetectFreq;
    float MinDetectPeak;
    float* d_float;
    cufftComplex* d_complex;
    float* d_amplitude;
    float* H_maxmagfreq;
    float* d_maxmagfreq;
    float* channelData;
    float* H_channelData;

    dim3 ThreadsPerBlock_0;
    dim3 BlocksPerGrid_0;
    dim3 th;
    dim3 b;
};
__global__ void Cvt2Complex(float* matrix_in, cufftComplex* matrix_out);
__global__ void getamplitude(cufftComplex* fftresult, float* sign_mag);
__global__ void ChannelSelect(float* Amps,float *channelData,int ChannelID);
__global__ void getPeakFreqs(float* signal_mag, float freq_samp, float* freq_mag);


#endif // SOURCEFINDING_H
