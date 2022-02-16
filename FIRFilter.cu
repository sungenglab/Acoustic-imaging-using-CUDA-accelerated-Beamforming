#include "FIRFilter.h"
#include <math.h>
#include <iostream>
#include "stdio.h"

using namespace std;

static const int   inputlen     =1024*32;
static const int   FIRlen       =1024;
static const int   Row_toeplitz =1024;
static const int   Col_input    =32;
static const int   Col_toeplitz =1024;
static const float alpha        =1.0f;
static const float beta         =0.0f;
static const float PI    =3.141592654f;

__device__ float static_input[inputlen];
__device__ float static_output[inputlen];
__device__ float static_coeff[FIRlen];
__device__ float static_toeplitz[Row_toeplitz*Col_toeplitz];


FIRFilter::FIRFilter():
    threads(32,32),
    blocks(1024/threads.x,1024/threads.y)
{
    checkCudaErrors(cudaMallocHost((void**)&m_coeff, sizeof(float)*FIRlen));
    checkCudaErrors(cublasCreate(&MatrixMultiply));

    checkCudaErrors(cudaEventCreate(&start_GPU));
    checkCudaErrors(cudaEventCreate(&stop_GPU));
}
FIRFilter::~FIRFilter()
{
//    free(m_coeff);
    checkCudaErrors(cudaFreeHost(m_coeff));
    checkCudaErrors(cublasDestroy(MatrixMultiply));

    checkCudaErrors(cudaEventDestroy(start_GPU));
    checkCudaErrors(cudaEventDestroy(stop_GPU));
}
float FIRFilter::window(int window_type, int n, int i)
{
    int k;
    float w= 1.0f;

    switch(window_type)
    {
    case 1://矩形窗
    {
        w = 1.0;
        break;
    }
    case 2://图基窗
    {
        k = (n - 2) / 10;
        if (i <= k)
            w = static_cast < float >(0.5 * (1.0 - cosf(i * PI / (k + 1))));
        if (i > n-k-2)
        {
            w = static_cast < float >(0.5 * (1.0 - cosf((n - i - 1) * PI / (k + 1))));
        }
        break;
    }
    case 3://三角窗
    {
        w = static_cast < float >(1.0 - fabs(1.0 - 2 * i / (n - 1.0)));
        break;
    }
    case 4://汉宁窗
    {
        w = static_cast < float >(0.5 * (1.0 - cosf( 2 * i * PI / (n - 1))));
        break;
    }
    case 5://海明窗
    {
        w = static_cast < float >(0.54 - 0.46 * cosf(2 * i * PI / (n - 1)));
        break;
    }
    case 6://布莱克曼窗
    {
        w = static_cast < float >(0.42 - 0.5 * cosf(2 * i * PI / (n - 1)) + 0.08 * cosf(4 * i * PI / (n - 1)));
        break;
    }
    }
    return w;
}

void FIRFilter::FIR_Coefficient(int order, int Filter_type, int window_type, int fs,float* h, float fln, float fhn)
{
    int i;
    int n2;
    int mid;
    float s;
    float wc1;
    float wc2;
    float delay;

    if ((order%2) == 0)/*如果阶数order是偶数*/
    {
        n2 = (order / 2) - 1;/**/
        mid = 1;//
    }
    else
    {
        n2 = order / 2;//order是奇数,则窗口长度为偶数
        mid = 0;
    }

    delay = static_cast < float >(order / 2.0);
    wc1 = 2 * PI * fln;
    wc2 = 2 * PI * fhn;

    switch (Filter_type)
    {
    case 1:
    {
        for(i=0; i<=n2; ++i)
        {
            s = i - delay;
            h[i] = (sinf(wc1 * s / fs) / (PI * s)) * window(window_type, order+1, i);//低通,窗口长度=阶数+1，故为order+1
            h[order - i] = h[i];
        }
        if (mid == 1)
        {
            h[order / 2] = wc1 / PI;//order为偶数时，修正中间值系数
        }
        break;
    }
    case 2://带通
    {
        for (i=0; i<=n2; i++)
        {
            s = i - delay;
            h[i] = (sinf(wc2 * s / fs) - sinf(wc1 * s / fs)) / (PI * s);
            h[i] = h[i] * window(window_type, order+1, i);
            h[order-i] = h[i];
        }
        if (mid == 1)
        {
            h[order / 2] = (wc2 - wc1) / PI;
        }
        break;
    }
    case 3://高通
    {
        for(i=0; i<=n2; i++)
        {
            s = i - delay;
            h[i] = (sinf(PI * s) - sinf(wc1 * s / fs)) / (PI * s);
            h[i] = h[i] * window(window_type, order+1, i);
            h[order-i] = h[i];
        }
        if (mid == 1)
        {
            h[order / 2] =static_cast < float >(1.0 - wc1 / PI) ;
        }
        break;
    }
    }

}
void FIRFilter::LowPassFilter(int order,int window_type,int fs,float fp,float* input,float* output)
{

    FIR_Coefficient(order,1,window_type,fs,m_coeff,fp,fp);

    //Call GPU function to calculate
    Filter_Cal(m_coeff,order,input,output);
}

void FIRFilter::BandPassFilter(int order,int window_type,int fs,float fp1,float fp2,float* input,float* output)
{
    checkCudaErrors(cudaEventRecord(start_GPU, 0));

    FIR_Coefficient(order,2,window_type,fs,m_coeff,fp1,fp2);

    //Call GPU function to calculate
    Filter_Cal(m_coeff,order,input,output);

    checkCudaErrors(cudaEventRecord(stop_GPU, 0));
    checkCudaErrors(cudaEventSynchronize(start_GPU));
    checkCudaErrors(cudaEventSynchronize(stop_GPU));
    checkCudaErrors(cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU));
    printf("The time for filtering: %fms\n", time_GPU);
}

void FIRFilter::HighPassFilter(int order,int window_type,int fs,float fp,float* input,float* output)
{
    FIR_Coefficient(order,3,window_type,fs,m_coeff,fp,fp);

    //Call GPU function to calculate
    Filter_Cal(m_coeff,order,input,output);

}
void FIRFilter::Filter_Cal(float* coeff,int order,float* input,float* output)
{
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_coeff, static_coeff));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_input, static_input));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_output, static_output));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_toeplitz, static_toeplitz));

    checkCudaErrors(cudaMemcpy(d_coeff,coeff,sizeof(float)*(order+1),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_input,input,sizeof(float)*inputlen,cudaMemcpyHostToDevice));

    Toeplitz<<<blocks,threads>>>(d_coeff,d_toeplitz,order);
    cudaError_t error0 = cudaGetLastError();
    if(!(error0==cudaSuccess))
        printf("kernel Toeplitz: %s\n", cudaGetErrorString(error0));

    checkCudaErrors(cublasSgemm(MatrixMultiply, CUBLAS_OP_N, CUBLAS_OP_N,
                                Col_input, Row_toeplitz, Col_toeplitz,
                                &alpha, d_input, Col_input, d_toeplitz,
                                Col_toeplitz, &beta, d_output, Col_input));

    checkCudaErrors(cudaMemcpy(output,d_output,sizeof(float)*inputlen,cudaMemcpyDeviceToHost));

}

__global__ void Toeplitz(float* coeff,float* coeff_Toeplitz,int order)
{
    unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int threadID =i*1024+j;

    if (threadID < 1024*1024)
    {
        int m= i-j;
        if (m<0 || m>(order+1))
        {
            coeff_Toeplitz[threadID]=0;
        }
        else
        {
            coeff_Toeplitz[threadID]=coeff[m];
        }
    }
}

