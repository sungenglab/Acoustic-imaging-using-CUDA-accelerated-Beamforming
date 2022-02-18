#include "SoundFieldCal.h"
#include "iostream"
#include <unordered_map>

using namespace std;

__device__ float static_oridata[ARRAY_CHANNEL_NUM * datalen];
__device__ cufftComplex static_complex[ARRAY_CHANNEL_NUM * datalen];
__device__ cufftComplex static_expects[ARRAY_CHANNEL_NUM * datalen];
__device__ cufftComplex static_transposed[ARRAY_CHANNEL_NUM * datalen];
__device__ cufftComplex static_covariance[ARRAY_CHANNEL_NUM * ARRAY_CHANNEL_NUM];
__device__ float static_amplitude[ARRAY_CHANNEL_NUM * datalen];
__device__ float static_maxmagfreq[ARRAY_CHANNEL_NUM];
__device__ float static_max_vector[Y_RESOLUTION];
__device__ float static_min_vector[Y_RESOLUTION];
__device__ int static_CoordXmax[Y_RESOLUTION];
__constant__ float static_plane_x[X_RESOLUTION];
__constant__ float static_plane_y[Y_RESOLUTION];
__constant__ float static_Array_x[ARRAY_CHANNEL_NUM];
__constant__ float static_Array_y[ARRAY_CHANNEL_NUM];
__device__   float static_SoundMatrix[X_RESOLUTION * Y_RESOLUTION];
__device__   float SingleResized[X_RESOLUTION * Y_RESOLUTION];
__device__   float static_SoundCentral[2];

__constant__ float static_plane_x1[X_PIXEL];
__constant__ float static_plane_y1[Y_PIXEL];

const float SoundFieldCal::ArrayPositon_X[ARRAY_CHANNEL_NUM] =
{
    -24.0384f, 39.7509f, 13.8076f, -6.7664f,-38.1862f, 61.4564f, 63.4630f, 60.3642f,
    46.8006f, 39.8480f, 17.7430f, 40.7046f, 35.2944f, 25.0836f,  1.3854f, 32.5728f,
    10.9574f, 11.5798f,-23.6358f, 13.2726f,-11.7486f, -8.3936f,-28.5104f,-26.1990f,
    -16.2422f,-36.8104f,-39.8642f,-51.0402f,-54.6978f,-63.3786f,-70.2680f,-26.4276f
};
const float SoundFieldCal::ArrayPositon_Y[ARRAY_CHANNEL_NUM] =
{
    -66.0538f,-62.3824f,-62.1676f,-51.0678f,-45.4798f,-28.6358f, 32.9846f,  7.9910f,
    -10.4240f,-39.8826f,-38.0846f, 34.1530f, 12.8932f,-15.9358f,-24.9528f, 56.3764f,
    23.1786f, 63.9546f, 17.2820f, 42.0778f, 58.6840f, 31.3590f, 36.5152f, -4.7598f,
    -31.2012f,15.2635f,-22.9716f, 35.7278f,  2.403f, -19.9466f, 18.8114f, 63.7948f
};
const float SoundFieldCal::ArrayPositon_Z[ARRAY_CHANNEL_NUM] =
{
    0.00f,    0.00f,     0.00f,  0.00f,    0.00f,     0.00f,    0.00f,    0.00f,
    0.00f,    0.00f,     0.00f,  0.00f,    0.00f,     0.00f,   0.00f,    0.00f,
    0.00f,    0.00f,     0.00f,  0.00f,    0.00f,     0.00f,    0.00f,    0.00f,
    0.00f,    0.00f,     0.00f,  0.00f,    0.00f,     0.00f,    0.00f,    0.00f
};

SoundFieldCal::SoundFieldCal() :
    m_sampfreq(70000.0f),
    m_sampdist(3.5f),
    m_velocity(340.0f),
    m_datalen(32 * 1024),
    p_ref(0.000000000001f),
    bottom_noise(0.01f),
    MinDetectFreq(2000.0f),
    central_SPL (0.0f),
    alpha{1.0f,0.0f},
    beta{0.0f,0.0f},


    ThreadsPerBlock_0(32, 32),
    BlocksPerGrid_0(Col / ThreadsPerBlock_0.x, Row / ThreadsPerBlock_0.y),

    ThreadsPerBlock_1(32, 1),
    BlocksPerGrid_1(Col / ThreadsPerBlock_1.x, Row / ThreadsPerBlock_1.y),

    blocks(Col / 32, Col / 32),
    threads(32, 32),

    ThreadsPerBlock_2(32, 5),
    BlocksPerGrid_2(X_RESOLUTION / ThreadsPerBlock_2.x, Y_RESOLUTION / ThreadsPerBlock_2.y),

    th(1, 1024),
    b(32, 1)
{
    checkCudaErrors(cudaMallocHost((void**)&m_soundMatrix, X_RESOLUTION * Y_RESOLUTION * sizeof(float)));

    InitArray();
    InitPlane();
    InitHandle();

    checkCudaErrors(cudaMallocHost((void**)&H_max_vector, Y_RESOLUTION * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&H_min_vector, Y_RESOLUTION * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&H_CoordXmax, Y_RESOLUTION * sizeof(int)));
    checkCudaErrors(cudaMallocHost((void**)&sound_central, 2 * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&d_meanAmp, ARRAY_CHANNEL_NUM * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&h_meanAmp, ARRAY_CHANNEL_NUM * sizeof(float)));

    checkCudaErrors(cudaMallocHost((void**)&H_maxmagfreq, ARRAY_CHANNEL_NUM * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&channelData, 1024 * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&H_channelData, 1024 * sizeof(float)));

    MyFilter= new FIRFilter();
}
SoundFieldCal::~SoundFieldCal()
{
    checkCudaErrors(cudaFreeHost(m_soundMatrix));

    checkCudaErrors(cudaFreeHost(m_Array_X));
    checkCudaErrors(cudaFreeHost(m_Array_Y));
    checkCudaErrors(cudaFreeHost(m_Plane_X));
    checkCudaErrors(cudaFreeHost(m_Plane_Y));
    checkCudaErrors(cudaFreeHost(m1_Plane_X));
    checkCudaErrors(cudaFreeHost(m1_Plane_Y));

    m_Plane_X = nullptr;
    m_Plane_Y = nullptr;
    m_Array_X = nullptr;
    m_Array_Y = nullptr;
    m_Array_Z = nullptr;

    checkCudaErrors(cudaEventDestroy(start_GPU));
    checkCudaErrors(cudaEventDestroy(stop_GPU));
    checkCudaErrors(cufftDestroy(PlanManyC2C));
    checkCudaErrors(cublasDestroy(CompMatrixMultiply));
    checkCudaErrors(cudaFreeHost(H_min_vector));
    checkCudaErrors(cudaFreeHost(H_max_vector));
    checkCudaErrors(cudaFreeHost(H_CoordXmax));
    checkCudaErrors(cudaFreeHost(sound_central));

    checkCudaErrors(cudaFree(d_meanAmp));
    checkCudaErrors(cudaFreeHost(h_meanAmp));
    checkCudaErrors(cudaFreeHost(H_maxmagfreq));
    checkCudaErrors(cudaFree(channelData));
    checkCudaErrors(cudaFreeHost(H_channelData));

    delete MyFilter;
}
void SoundFieldCal::InitArray()
{
    checkCudaErrors(cudaMallocHost((void**)&m_Array_X, ARRAY_CHANNEL_NUM * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&m_Array_Y, ARRAY_CHANNEL_NUM * sizeof(float)));

    for (int i = 0; i < ARRAY_CHANNEL_NUM; i++)
    {
        m_Array_X[i] = static_cast <float>(ArrayPositon_X[i] / 1000);
        m_Array_Y[i] = static_cast <float>(ArrayPositon_Y[i] / 1000);
    }
}

void SoundFieldCal::InitPlane()
{
    checkCudaErrors(cudaMallocHost((void**)&m_Plane_X, X_RESOLUTION * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&m_Plane_Y, Y_RESOLUTION * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&m1_Plane_X, X_PIXEL * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&m1_Plane_Y, Y_PIXEL * sizeof(float)));

    float theta = 45.8f;
    float eta = static_cast <float>(m_sampdist * tanf(0.5f * theta * PI / 180.0f) * 0.5f);

    float xRange = 4 * eta;
    float yRange = 3 * eta;

    float f_dx = xRange / (X_RESOLUTION - 1);
    float f_dy = yRange / (Y_RESOLUTION - 1);

    float f1_dx = xRange / (X_PIXEL - 1);
    float f1_dy = yRange / (Y_PIXEL - 1);

    for (int i = 0; i < X_RESOLUTION; i++)
    {
        m_Plane_X[i] = -xRange / 2 + i * f_dx;
    }
    for (int i = 0; i < Y_RESOLUTION; i++)
    {
        m_Plane_Y[i] = -yRange / 2 + i * f_dy;
    }

    for (int i = 0; i < X_PIXEL; i++)
    {
        m1_Plane_X[i] = -xRange / 2 + i * f1_dx;
    }
    for (int i = 0; i < Y_PIXEL; i++)
    {
        m1_Plane_Y[i] = -yRange / 2 + i * f1_dy;
    }
}
void SoundFieldCal::InitHandle()
{
    int BATCH = Col;
    int rank = 1;
    int n1[1] = { Row };
    int istride = Col;
    int idist = 1;
    int ostride = Col;
    int odist = 1;
    int inembed[2] = { Col, Row };
    int onembed[2] = { Col, Row };

    checkCudaErrors(cufftPlanMany(&PlanManyC2C, rank, n1, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, BATCH));
    checkCudaErrors(cublasCreate(&CompMatrixMultiply));

    checkCudaErrors(cudaEventCreate(&start_GPU));
    checkCudaErrors(cudaEventCreate(&stop_GPU));
}
float SoundFieldCal::SPL_cal(float max_power)
{
    float  power_level=10.0f *log10f(max_power /p_ref);
    float  SPL =power_level-10.0f*log10f(TwoPI*m_sampdist);
    return SPL;
}
cufftComplex* SoundFieldCal::Covariance_cal(float SourceFreq)
{
    float fp1=SourceFreq-1000.0f;
    float fp2=SourceFreq+1000.0f;
    if(fp1<0.0f || fp2>(m_sampfreq*0.5f))
    {
       fp1=0.0f;
       fp2=m_sampfreq/2.0f;
    }
    float* m_filtered;
    checkCudaErrors(cudaMallocHost((void**)&m_filtered, m_datalen * sizeof(float)));
    MyFilter->BandPassFilter(1023,1,int(m_sampfreq),fp1,fp2,m_oridata,m_filtered);

    checkCudaErrors(cudaMemcpy(d_float, m_filtered,
                               sizeof(float) * m_datalen,
                               cudaMemcpyHostToDevice));

    Real2Complex << < BlocksPerGrid_0, ThreadsPerBlock_0 >> > (d_float, d_complex);

    checkCudaErrors(cufftExecC2C(PlanManyC2C, d_complex, d_complex, CUFFT_FORWARD));

    data_revise << < BlocksPerGrid_1, ThreadsPerBlock_1 >> > (d_complex, d_complex);
    cudaError_t error4 = cudaGetLastError();
    if (!(error4 == cudaSuccess))
        printf("kernel data_revise: %s\n", cudaGetErrorString(error4));

    checkCudaErrors(cufftExecC2C(PlanManyC2C, d_complex, d_complex, CUFFT_INVERSE));

    data_recovery << < BlocksPerGrid_0, ThreadsPerBlock_0 >> > (d_complex, d_float, d_complex);
    cudaError_t error5 = cudaGetLastError();
    if (!(error5 == cudaSuccess))
        printf("kernel data_recovery: %s\n", cudaGetErrorString(error5));

    GetExpectedValue << < b, th >> > (d_complex, d_expects);
    cudaError_t error6 = cudaGetLastError();
    if (!(error6 == cudaSuccess))
        printf("kernel GetExpectedValue: %s\n", cudaGetErrorString(error6));

    Matrix_Subtraction << <BlocksPerGrid_0, ThreadsPerBlock_0 >> > (d_complex, d_expects, d_complex);
    cudaError_t error7 = cudaGetLastError();
    if (!(error7 == cudaSuccess))
        printf("kernel Matrix_Subtraction: %s\n", cudaGetErrorString(error7));

    Matrix_Conj_Transpose << <BlocksPerGrid_0, ThreadsPerBlock_0 >> > (d_complex, d_transposed);
    cudaError_t error8 = cudaGetLastError();
    if (!(error8 == cudaSuccess))
        printf("kernel Matrix_Conj_Transpose: %s\n", cudaGetErrorString(error8));

    checkCudaErrors(cublasCgemm3m(CompMatrixMultiply,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k, &alpha,
                                    d_complex, n,
                                    d_transposed, k, &beta,
                                    d_covariance, n));

    Get_Cov << <BlocksPerGrid_0, ThreadsPerBlock_0 >> > (d_covariance, (Row - 1));
    cudaError_t error9 = cudaGetLastError();
    if (!(error9 == cudaSuccess))
        printf("kernel Get_Cov: %s\n", cudaGetErrorString(error9));

    checkCudaErrors(cudaFreeHost(m_filtered));
    return d_covariance;
}
void  SoundFieldCal::calculate()
{
    checkCudaErrors(cudaEventRecord(start_GPU, 0));

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_float, static_oridata));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_complex, static_complex));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_expects, static_expects));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_transposed, static_transposed));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_covariance, static_covariance));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_amplitude, static_amplitude));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_maxmagfreq, static_maxmagfreq));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_sound_field, static_SoundMatrix));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_max_vector, static_max_vector));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_min_vector, static_min_vector));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_CoordXmax, static_CoordXmax));

    checkCudaErrors(cudaMemcpy(d_float, m_oridata, sizeof(float) * ARRAY_CHANNEL_NUM * datalen, cudaMemcpyHostToDevice));

    Real2Complex << < BlocksPerGrid_0, ThreadsPerBlock_0 >> > (d_float, d_complex);
    cudaError_t error0 = cudaGetLastError();
    if (!(error0 == cudaSuccess))
        printf("kernel Real2Complex: %s\n", cudaGetErrorString(error0));

    //FFT
    checkCudaErrors(cufftExecC2C(PlanManyC2C, d_complex, d_complex, CUFFT_FORWARD));
    //    cudaDeviceSynchronize();

    getSignal_amplitude << <BlocksPerGrid_0, ThreadsPerBlock_0>> > (d_complex, d_amplitude);
    cudaError_t error1 = cudaGetLastError();
    if (!(error1 == cudaSuccess))
        printf("kernel getSignal_amplitude: %s\n", cudaGetErrorString(error1));

    getmeanAmp<<<b, th>>>(d_amplitude,d_meanAmp);
    cudaError_t error2 = cudaGetLastError();
    if (!(error2 == cudaSuccess))
        printf("kernel getmeanAmp: %s\n", cudaGetErrorString(error2));

    checkCudaErrors(cudaMemcpy(h_meanAmp, d_meanAmp, ARRAY_CHANNEL_NUM * sizeof(float), cudaMemcpyDeviceToHost));

    meanAmp=get_mean(h_meanAmp,ARRAY_CHANNEL_NUM);
//    cout<<"meanAmp:"<<meanAmp<<endl;

    //get Peak Freqency
    getPeakFreq << <b, th >> > (d_amplitude, m_sampfreq, d_maxmagfreq);
    cudaError_t error3 = cudaGetLastError();
    if (!(error3 == cudaSuccess))
        printf("kernel getPeakFreq: %s\n", cudaGetErrorString(error3));

    checkCudaErrors(cudaMemcpy(H_maxmagfreq, d_maxmagfreq, ARRAY_CHANNEL_NUM * sizeof(float), cudaMemcpyDeviceToHost));

    maxfreqChannel = top1Frequent(H_maxmagfreq,ARRAY_CHANNEL_NUM);

    float meanFreq = get_mean(H_maxmagfreq, ARRAY_CHANNEL_NUM);

    if ((meanFreq - 1000.f) < 0.0f|| (meanAmp-bottom_noise) <0.0f)
    {
        memset(m_soundMatrix, 0, sizeof(float) * X_RESOLUTION * Y_RESOLUTION);
        central_SPL =0.0f;
        return;
    }
    else
    {
        cout<<"channel:"<<maxfreqChannel->Position_Y<<",Freqence:"<<maxfreqChannel->value<<endl;
        ChannelChoose<<<b, th>>>(d_amplitude,channelData,maxfreqChannel->Position_Y);

        checkCudaErrors(cudaMemcpy(H_channelData, channelData, 1024 * sizeof(float), cudaMemcpyDeviceToHost));

        SourceDetect=PeaksFinding(H_channelData);

        cout<<"Source Detected :"<<SourceDetect->SourceNums<<endl;
        for(int i=0;i<SourceDetect->SourceNums;i++)
        {
            cout<<"source"<<i<<": "<<SourceDetect->SourceFreq[i]<<" Hz"<<endl;
        }

        checkCudaErrors(cudaGetSymbolAddress((void**)&d_array_x, static_Array_x));
        checkCudaErrors(cudaGetSymbolAddress((void**)&d_array_y, static_Array_y));
        checkCudaErrors(cudaGetSymbolAddress((void**)&d_plane_x, static_plane_x));
        checkCudaErrors(cudaGetSymbolAddress((void**)&d_plane_y, static_plane_y));

        checkCudaErrors(cudaGetSymbolAddress((void**)&d1_plane_x, static_plane_x1));
        checkCudaErrors(cudaGetSymbolAddress((void**)&d1_plane_y, static_plane_y1));

        checkCudaErrors(cudaMemcpy(d_array_x, m_Array_X, sizeof(float) * ARRAY_CHANNEL_NUM, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_array_y, m_Array_Y, sizeof(float) * ARRAY_CHANNEL_NUM, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_plane_x, m_Plane_X, sizeof(float) * X_RESOLUTION, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_plane_y, m_Plane_Y, sizeof(float) * Y_RESOLUTION, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(d1_plane_x, m1_Plane_X, sizeof(float) * X_PIXEL, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d1_plane_y, m1_Plane_Y, sizeof(float) * Y_PIXEL, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaGetSymbolAddress((void**)&d_sound_central, static_SoundCentral));
        checkCudaErrors(cudaGetSymbolAddress((void**)&d_singleResized, SingleResized));

        dim3 ttt(32, 1);
        dim3 bbb(X_RESOLUTION / ttt.x, Y_RESOLUTION/ ttt.y);
        dim3 mt(640,1);
        dim3 mb(1,480);
        float  max_value = 0.0f;
        float  min_value = 0.0f;

        zeros<<<BlocksPerGrid_2, ThreadsPerBlock_2>>>(d_sound_field);
        if(SourceDetect->SourceNums==0)
        {
            memset(m_soundMatrix, 0, sizeof(float) * X_RESOLUTION * Y_RESOLUTION);
            central_SPL =0.0f;
            return;
        }
        else if(SourceDetect->SourceNums>0)
        {
            for(int i=0;i<SourceDetect->SourceNums;i++)
            {
                cufftComplex* cov=nullptr;
                cov=Covariance_cal(SourceDetect->SourceFreq[i]);

                Beamforming << <bbb, ttt>> > (d_plane_x, d_plane_y,
                                              d_array_x, d_array_y,
                                              m_velocity,SourceDetect->SourceFreq[i] , m_sampdist,
                                              cov, d_singleResized,d_sound_central);

                cudaError_t error11 = cudaGetLastError();
                if (!(error11 == cudaSuccess))
                    printf("kernel Beamforming: %s\n", cudaGetErrorString(error11));
                if(i==0)
                {
                    checkCudaErrors(cudaMemcpy(sound_central, d_sound_central,
                                               2 * sizeof(float),
                                               cudaMemcpyDeviceToHost));
                    //linear interpolation to calculate central SPL
                    float central_sound = sound_central[1] + (sound_central[0]-sound_central[1]) *0.5f;
                    central_SPL =SPL_cal(central_sound);
                }
                //Separate normalization
                Matrix_Max << < mb, mt >> > (d_singleResized, d_max_vector,d_CoordXmax);
                Matrix_Min << < mb, mt >> > (d_singleResized, d_min_vector);
                checkCudaErrors(cudaMemcpy(H_max_vector, d_max_vector,
                                           Y_RESOLUTION * sizeof(float),
                                           cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(H_min_vector, d_min_vector,
                                           Y_RESOLUTION * sizeof(float),
                                           cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(H_CoordXmax, d_CoordXmax,
                                           Y_RESOLUTION * sizeof(float),
                                           cudaMemcpyDeviceToHost));

                Max = max1(H_max_vector, H_CoordXmax,Y_RESOLUTION);
                min_value = min(H_min_vector,Y_RESOLUTION);
//                cout<<"max_value: "<<Max->value<<endl;
//                cout<<"min_value: "<<min_value<<endl;
                cout<<"Source Position:("<<Max->Position_X<<","<<Max->Position_Y<<")"<<endl;

                //Find Max SoundPower and return its coordinate
                SourceDetect->SPL[i]=SPL_cal(Max->value);
                SourceDetect->Position_X[i]=Max->Position_X;
                SourceDetect->Position_Y[i]=Max->Position_Y;

                Normollizing << <BlocksPerGrid_2, ThreadsPerBlock_2 >> > (d_singleResized, Max->value,min_value,0.0f);

                //Add up normalization results
                MatAdd<< <BlocksPerGrid_2, ThreadsPerBlock_2 >> >(d_sound_field,d_singleResized);
                cudaError_t error13 = cudaGetLastError();
                if (!(error13 == cudaSuccess))
                    printf("MatAdd: %s\n", cudaGetErrorString(error13));
            }
        }

        Matrix_Max << < mb, mt  >> > (d_sound_field, d_max_vector,d_CoordXmax);
        Matrix_Min << < mb, mt  >> > (d_sound_field, d_min_vector);

        checkCudaErrors(cudaMemcpy(H_max_vector, d_max_vector, Y_RESOLUTION * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(H_min_vector, d_min_vector, Y_RESOLUTION * sizeof(float), cudaMemcpyDeviceToHost));

        max_value = max(H_max_vector, Y_RESOLUTION);
        min_value = min(H_min_vector, Y_RESOLUTION);

        Normollizing << <BlocksPerGrid_2, ThreadsPerBlock_2 >> > (d_sound_field, max_value,min_value,ImageThreshold);

        RemoveEdge << <BlocksPerGrid_2, ThreadsPerBlock_2 >> > (d_sound_field, ImageThreshold);
        cudaError_t error14 = cudaGetLastError();
        if (!(error14 == cudaSuccess))
            printf("kernel RemoveEdge: %s\n", cudaGetErrorString(error14));

        checkCudaErrors(cudaMemcpy(m_soundMatrix, d_sound_field,
                                   Y_RESOLUTION * X_RESOLUTION * sizeof(float),
                                   cudaMemcpyDeviceToHost));
    }

    checkCudaErrors(cudaEventRecord(stop_GPU, 0));
    checkCudaErrors(cudaEventSynchronize(start_GPU));
    checkCudaErrors(cudaEventSynchronize(stop_GPU));
    checkCudaErrors(cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU));
    cout<<"The time for GPU: "<<time_GPU<<"ms"<<endl;

}

__global__ void Real2Complex(float* matrix_in, cufftComplex* matrix_out)
{
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadID < Col * Row)
    {
        matrix_out[threadID].x = matrix_in[threadID];
        matrix_out[threadID].y = 0.0f;
    }
}

__global__ void data_recovery(cufftComplex* matrix_in, float* original, cufftComplex* matrix_out)
{
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadID < Col * Row)
    {
        matrix_out[threadID].x = original[threadID];
        matrix_out[threadID].y = matrix_in[threadID].y / Row;
    }
}

__global__ void data_revise(cufftComplex* matrix_in, cufftComplex* matrix_out)
{
    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (blockID < Row)
    {
        if (blockID == 0)
        {
            matrix_out[threadID].x = matrix_in[threadID].x;
            matrix_out[threadID].y = matrix_in[threadID].y;
        }
        else if ((0 < blockID) && (blockID < gridDim.y / 2))
        {
            matrix_out[threadID].x = 2.0f * matrix_in[threadID].x;
            matrix_out[threadID].y = 2.0f * matrix_in[threadID].y;
        }
        else
        {
            matrix_out[threadID].x = 0.0f;
            matrix_out[threadID].y = 0.0f;
        }
    }
}

__global__ void GetExpectedValue(cufftComplex* matrix_in, cufftComplex* matrix_out)
{
    //global coordinate
    unsigned int i = threadIdx.y;
    unsigned int j = blockIdx.x;
    const int blocksize=1024;

    //global index
    unsigned int threadId = i * 32 + j;

    __shared__ cufftComplex reduced[1024];

    //write to shared memory
    reduced[i] = matrix_in[threadId];
    __syncthreads();

    //以下采用折半归约求每列和
    for (unsigned int offset = blocksize >>1; offset > 0; offset >>= 1)
    {
        if (i < offset)
        {
            reduced[i].x += reduced[i + offset].x;
            reduced[i].y += reduced[i + offset].y;
        }
        __syncthreads();
    }
    // 取均值
    if (i < 1024)
    {
        reduced[i].x =static_cast<float>(reduced[i].x / blocksize) ;
        reduced[i].y =static_cast<float>(reduced[i].y / blocksize) ;
    }
    __syncthreads();
    reduced[i]=reduced[0];
    __syncthreads();
    matrix_out[threadId] = reduced[i];
}

__global__ void Matrix_Subtraction(cufftComplex* matrix_A, cufftComplex* matrix_B, cufftComplex* matrix_C)
{
    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadID < Col * Row)
    {
        matrix_C[threadID].x = matrix_A[threadID].x - matrix_B[threadID].x;
        matrix_C[threadID].y = matrix_A[threadID].y - matrix_B[threadID].y;
    }
}
__global__ void Matrix_Transpose(cufftComplex* matrix_A, cufftComplex* matrix_B)
{

    const unsigned int K = 32;
    unsigned int in_corner_i = blockIdx.x * K, in_corner_j = blockIdx.y * K;
    unsigned int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

    unsigned int x = threadIdx.x;
    unsigned int y = threadIdx.y;

    __shared__ cufftComplex tile[K][K];

    tile[y][x] = matrix_A[(in_corner_i + x) + (in_corner_j + y) * Col];
    __syncthreads();

    matrix_B[(out_corner_i + x) + (out_corner_j + y) * Col] = tile[x][y];

}
__global__ void Matrix_Conj_Transpose(cufftComplex* matrix_A, cufftComplex* matrix_B)
{

    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    unsigned int threadID_Transposed = threadIdx.x * blockDim.y * gridDim.y + blockID * blockDim.y + threadIdx.y;

    if (threadID < Col * Row)
    {
        matrix_B[threadID_Transposed].x =  matrix_A[threadID].x;
        matrix_B[threadID_Transposed].y = -matrix_A[threadID].y;
    }
    //matrix_A read only,可采用非合并全局访存
    //matrix_B 没有缓存，采用合并访存模式
}

__global__ void Get_Cov(cufftComplex* matrix_A, float datasize)
{
    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadID < Col * Row)
    {
        matrix_A[threadID].x = matrix_A[threadID].x / datasize;
        matrix_A[threadID].y = matrix_A[threadID].y / datasize;
    }
}

__device__ cufftComplex Complex_Multi(cufftComplex A, cufftComplex B)
{
    cufftComplex C;
    C.x = (float)(A.x * B.x - A.y * B.y);
    C.y = (float)(A.x * B.y + A.y * B.x);
    return C;
}
__device__ cufftComplex Complex_Add(cufftComplex A, cufftComplex B)
{
    cufftComplex C;
    C.x = (float)(A.x + B.x);
    C.y = (float)(A.y + B.y);
    return C;
}

__global__ void Beamforming(const float* x, const float* y,
                            const float* px, const float* py,
                            const float Velocity, const float freq,
                            const float Radial_distance, const cufftComplex* Rx,
                            float* sound_field ,float* sound_central)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;

    float distance = 0.0f;
    cufftComplex steer_vector[ARRAY_CHANNEL_NUM];
    cufftComplex temp_vector[ARRAY_CHANNEL_NUM];
    cufftComplex soundcomplex{0.0f,0.0f};
    __shared__ cufftComplex RX[ARRAY_CHANNEL_NUM];

    for (int k = 0; k < ARRAY_CHANNEL_NUM; k++)
    {
        distance = sqrtf((x[i] - px[k])*(x[i] - px[k]) + (y[j] - py[k])*(y[j] - py[k]) + Radial_distance*Radial_distance);

        steer_vector[k].x = cosf(distance * DoublePI * freq / Velocity);
        steer_vector[k].y = sinf(distance * DoublePI * freq / Velocity) * (-1);
    }
    for (int k1 = 0; k1 < ARRAY_CHANNEL_NUM; k1++)
    {
        RX[threadIdx.x].x=Rx[k1*32+threadIdx.x].x;
        RX[threadIdx.x].y=Rx[k1*32+threadIdx.x].y;

        temp_vector[k1].x = 0.0f;
        temp_vector[k1].y = 0.0f;

        for (int k3 = 0; k3 < ARRAY_CHANNEL_NUM; k3++)
        {
            temp_vector[k1].x += steer_vector[k3].x * RX[k3].x + steer_vector[k3].y * RX[k3].y;
            temp_vector[k1].y += steer_vector[k3].x * RX[k3].y - steer_vector[k3].y * RX[k3].x;
        }
    }

    for (int k = 0; k < ARRAY_CHANNEL_NUM; k++)
    {
        soundcomplex.x += temp_vector[k].x * steer_vector[k].x - temp_vector[k].y * steer_vector[k].y;
        soundcomplex.y += temp_vector[k].x * steer_vector[k].y + temp_vector[k].y * steer_vector[k].x;
    }
    sound_field [j * 640 + i] = 0.0f;

    sound_field[j * 640 + i] = sqrtf(soundcomplex.x *soundcomplex.x + soundcomplex.y*soundcomplex.y);
    sound_central[0] = sound_field[153280];
    sound_central[1] = sound_field[153919];
}

__global__ void Normollizing(float* matrix, float max_value, float min_value,float threshold)
{
    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadID < X_RESOLUTION * Y_RESOLUTION)
    {
        matrix[threadID] = (matrix[threadID]-min_value) / (max_value-min_value);
        if (matrix[threadID] < threshold)
        {
            matrix[threadID] = 0;
        }
    }
}

__global__ void RemoveEdge(float* matrix, float threshold)
{
    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadID < X_RESOLUTION * Y_RESOLUTION &&(!(matrix[threadID] == 0)))
    {
        matrix[threadID] = (matrix[threadID] - threshold) / (1 - threshold) * 255;
    }
}

__global__ void Matrix_Max(float* matrix, float* max_value,int* position_x)
{
    const int threadsPerBlock=640;
    __shared__ float partialMax[threadsPerBlock];
    __shared__ int idx[threadsPerBlock];

    //calculate index;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
    partialMax[i]=matrix[j*640+i];
    idx[i]=i;
    //传输同步
    __syncthreads();

    //在共享存储器中进行规约
    for (unsigned int stride = blockDim.x>>1; stride > 0; stride >>= 1)
    {
        if (i < stride && partialMax[i] - partialMax[i + stride]<0.0f)
        {
            partialMax[i] = partialMax[i + stride];
            idx[i]=idx[i + stride];
        }
    }
    __syncthreads();
    //将当前block的计算结果写回输出数组
    if (i == 0)
    {
        max_value[blockIdx.y] = partialMax[i];
        position_x[blockIdx.y] =idx[i];
    }
}

__global__ void Matrix_Min(float* matrix, float* min_value)
{
    //申请共享内存,存在于每个block中
    const int threadsPerBlock = 640;
    __shared__ float partialMin[threadsPerBlock];
    //确定索引
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;

    //传global memory数据到shared memory
    partialMin[i]=matrix[j*640+i];
    //传输同步
    __syncthreads();

    //在共享存储器中进行规约
    for (unsigned int stride = blockDim.x >>1; stride > 0; stride >>= 1)
    {
        if ( i<stride && partialMin[i] - partialMin[i + stride]>0.0f)
        {
            partialMin[i] = partialMin[i + stride];
        }
    }
    //将当前block的计算结果写回输出数组
    if (i == 0)
    {
        min_value[blockIdx.y] = partialMin[i];
    }
}

__host__ MaxMinValue* max1(float* vector, int* index_x, int size)
{
    MaxMinValue *max =new MaxMinValue;
    max->value=vector[0];
    max->Position_X=0;
    max->Position_Y=0;
    for (int i = 0; i < size; i++)
    {
        if (vector[i] - max->value > 0)
        {
            max->value = vector[i];
            max->Position_X =index_x[i];
            max->Position_Y =i;
        }
    }
    return max;
}
__host__  MaxMinValue* min1(float* vector, int* index_x, int size)
{
    MaxMinValue *min=new MaxMinValue;
    min->value=vector[0];
    min->Position_X=0;
    min->Position_Y=0;
    for (int i = 0; i < size; i++)
    {
        if (vector[i] - min->value<0)
        {
            min->value = vector[i];
            min->Position_X =index_x[i];
            min->Position_Y =i;
        }
    }
    return  min;
}
__host__ MaxMinValue* top1Frequent(float* vector, int size)
{
    MaxMinValue *max =new MaxMinValue;
    max->value=vector[0];
    max->Position_X=0;
    max->Position_Y=0;
    //Channel with the highest value
    if(0)
    {
        for (int i = 0; i < size; i++)
        {
            if (vector[i] - max->value > 0)
            {
                max->value = vector[i];
                max->Position_Y =i;
            }
        }
    }
    //Channel with the most frequent
    float index[]{0 ,1 ,2 ,3 ,4 ,5 ,6 ,7,
                  8 ,9 ,10,11,12,13,14,15,
                  16,17,18,19,20,21,22,23,
                  24,25,26,27,28,29,30,31};
    quickSort(0,size-1,vector,index);
    unordered_map<float, int> occurences;
    for(int i=0;i<size;i++)
    {
        occurences[vector[i]]++;
    }
    unordered_map<float,int>::iterator it=occurences.begin();
    int most=0,idx=0,channelID=0;
    while(it != occurences.end())
    {
        if((it->second)>most)
        {
            most=it->second;
        }
        idx+=it->second;
        if(it->second==most)
        {
            channelID=idx-most;
        }
//        cout<<it->first<<" "<<it->second<<endl;
        it++;
    }
    max->value=vector[channelID];
    max->Position_Y=static_cast<int>(index[channelID]);

    return max;
}

__global__ void getSignal_amplitude(cufftComplex* fftresult, float* sign_mag)
{
    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadID < 1024 * 32)
    {
        sign_mag[threadID] = sqrtf(fftresult[threadID].x * fftresult[threadID].x + fftresult[threadID].y * fftresult[threadID].y);
    }
}
__global__ void getPeakFreq(float* signal_mag, float freq_samp, float* freq_mag)
{
    unsigned int i = threadIdx.y;
    unsigned int j = blockIdx.x;
    const int blocksize=1024;

    __shared__ float AmpPerChannel[blocksize];
    __shared__ int index[blocksize];

    AmpPerChannel[i] = signal_mag[i * 32 + j];
    index[i]=i;
    //Remove DC component
    AmpPerChannel[0]=0.0f;
    __syncthreads();

    for (unsigned int stride = (blocksize/2)>>1; stride > 0; stride >>= 1)
    {
        if (i < stride && AmpPerChannel[i] - AmpPerChannel[i + stride]<0.0f)
        {
            AmpPerChannel[i] = AmpPerChannel[i + stride];
            index[i]=index[i + stride];
        }
    }
    __syncthreads();
    if(i==0)
    {
        freq_mag[j]=static_cast<float>(index[i]*freq_samp/1024) ;
    }
}
__host__ float get_sum(float* vector, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        sum += vector[i];
    }
    return sum;
}
__host__ float get_mean(float* vector, int size)
{
    float mean = 0.0f;
    float sum = 0.0f;
    sum = get_sum(vector, size);
    mean = sum / size;
    return mean;
}
__global__ void getmeanAmp(float* input,float* output)
{
    //global coordinate
    unsigned int i = threadIdx.y;
    unsigned int j = blockIdx.x;

    //global index
    unsigned int threadId = i * 32 + j;
    const int blocksize=1024;

    __shared__ float reduced[blocksize];

    //write to shared memory
    reduced[i] = input[threadId];
    __syncthreads();

    for (unsigned int offset = blocksize >>1; offset > 0; offset >>= 1)
    {
        if (i < offset)
        {
            reduced[i] += reduced[i + offset];
        }
        __syncthreads();
    }

    // 取均值
    if (i < 1024)
    {
        reduced[i] = static_cast<float>(reduced[i] / blocksize);
    }
    __syncthreads();

    if (i == 0)
    {
        output[j] = reduced[i];
    }
}
__global__ void MatAdd(float* matrix_A,float* matrix_B)
{
    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if(threadID<X_RESOLUTION*Y_RESOLUTION)
    {
        matrix_A[threadID]=matrix_A[threadID]+matrix_B[threadID];
    }
}

__global__ void MultiSources( float* x1,float* y1,
                              float* px,float* py,
                              float Velocity,float freq,
                              float Radial_distance,
                              cufftComplex* Rx,
                              float* sound_field,
                              float* sound_central)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;

    float distance = 0.0f;
    cufftComplex steer_vector[ARRAY_CHANNEL_NUM];
    cufftComplex temp_vector[ARRAY_CHANNEL_NUM];
    cufftComplex soundcomplex[X_PIXEL* Y_PIXEL];
    __shared__ cufftComplex Rx_vector[ARRAY_CHANNEL_NUM];

    for (int k = 0; k < ARRAY_CHANNEL_NUM; k++)
    {
        distance = sqrtf((x1[i] - px[k])*(x1[i] - px[k]) + (y1[j] - py[k])*(y1[j] - py[k]) + Radial_distance*Radial_distance);

        steer_vector[k].x = cosf(distance * DoublePI * freq / Velocity);
        steer_vector[k].y = sinf(distance * DoublePI * freq / Velocity) * (-1);
    }
    for (int k1 = 0; k1 < ARRAY_CHANNEL_NUM; k1++)
    {
        Rx_vector[threadIdx.x].x=Rx[k1*32+threadIdx.x].x;
        Rx_vector[threadIdx.x].y=Rx[k1*32+threadIdx.x].y;

        temp_vector[k1].x = 0.0f;
        temp_vector[k1].y = 0.0f;

        for (int k3 = 0; k3 < ARRAY_CHANNEL_NUM; k3++)
        {
            temp_vector[k1].x += steer_vector[k3].x * Rx_vector[k3].x + steer_vector[k3].y * Rx_vector[k3].y;
            temp_vector[k1].y += steer_vector[k3].x * Rx_vector[k3].y - steer_vector[k3].y * Rx_vector[k3].x;
        }
    }

    soundcomplex[j * X_PIXEL + i].x = 0.0f;
    soundcomplex[j * X_PIXEL + i].y = 0.0f;

    for (int k = 0; k < ARRAY_CHANNEL_NUM; k++)
    {
        soundcomplex[j * X_PIXEL + i].x += temp_vector[k].x * steer_vector[k].x - temp_vector[k].y * steer_vector[k].y;
        soundcomplex[j * X_PIXEL + i].y += temp_vector[k].x * steer_vector[k].y + temp_vector[k].y * steer_vector[k].x;
    }

    sound_field[j * X_PIXEL + i]= sqrtf(soundcomplex[j * X_PIXEL + i].x *soundcomplex[j * X_PIXEL + i].x + soundcomplex[j * X_PIXEL + i].y*soundcomplex[j * X_PIXEL + i].y);

    sound_central[0] = sound_field[38240];
    sound_central[1] = sound_field[38559];
}
__global__ void PixelResize(float* before,float* after)
{
    //before:X_PIXEL * Y_PIXEL
    //after :X_RESOLUTION * Y_RESOLUTION
    unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int zoom=X_RESOLUTION/X_PIXEL;

    if(j<X_RESOLUTION && i<Y_RESOLUTION)
    {
        int dx =j%zoom;
        int dy =i%zoom;

        float a=BoundCheck(before, j/zoom, i/zoom,X_PIXEL,Y_PIXEL);
        float b=BoundCheck(before, j/zoom+1, i/zoom,X_PIXEL,Y_PIXEL);
        float c=BoundCheck(before, j/zoom, i/zoom+1,X_PIXEL,Y_PIXEL);
        float d=BoundCheck(before, j/zoom+1, i/zoom+1,X_PIXEL,Y_PIXEL);

        after[i*X_RESOLUTION+j]=(1-dx)*(1-dy)*a +(1-dx)*dy*b+dx*(1-dy)*c+dx*dy*d;
    }
}
__device__ float BoundCheck(float* matrix,int x,int y,int xcol,int yrow)
{
    float value=0.0f;
    if((x>=xcol)||(x<0)||(y>=yrow)||(y<0))
        value=0.0f;
    else
        value=matrix[y*xcol+x];
    return value;
}
__global__ void ChannelChoose(float* Amps,float *channelData,int ChannelID)
{
    unsigned int i = threadIdx.y;
    channelData[i]=0.0f;
    channelData[i]=Amps[i * 32+ChannelID];

}
SourceInf* SoundFieldCal::PeaksFinding(float* channelData)
{
    int PeakFreq[512]{0};
    int valleyFreq[512]{0};
    float Peaks[512]{0};
    float valleys[512]{0};

    float smooth1[512]{0};
    float smooth2[512]{0};
    float vpd[512]{0};

    smooth1[0]=channelData[0];
    smooth1[511]=channelData[511];
    //Three points smoothing twice
    for(int i=1;i<511;i++)
    {
       smooth1[i]=(channelData[i-1]+channelData[i]+channelData[i+1])/3.0f;
    }
    for(int i=510;i>0;i--)
    {
       smooth2[i]=(smooth1[i-1]+smooth1[i]+smooth1[i+1])/3.0f;
    }
    smooth2[0]=channelData[0];
    smooth2[511]=channelData[511];

    //Find all Peaks and valleys,sampFreq*(i)/1024;
    int PeakIndex=0;
    int valleyIndex=0;
    for(int i=1;i<511;i++)
    {
        if((smooth2[i]-smooth2[i-1]>0.0f) && (smooth2[i]-smooth2[i+1]>0.0f))
        {
            Peaks[PeakIndex]=smooth2[i];
            PeakFreq[PeakIndex]=i;
            PeakIndex++;
        }
        if((smooth2[i]-smooth2[i-1]<0.0f) && (smooth2[i]-smooth2[i+1]<0.0f))
        {
            valleys[valleyIndex]=smooth2[i];
            valleyFreq[valleyIndex]=i;
            valleyIndex++;
        }
    }

    int pcount=PeakIndex;
    int vcount=valleyIndex;

    //boundary check
    if(pcount>2 && vcount>2)
    {
        if(PeakFreq[0]<valleyFreq[0])
            PeakIndex=1;
        else
            PeakIndex=0;
//        valleyIndex=1;
    }
    if (PeakIndex==2)
    {
        //discard first peak
        for(int i=0;i<PeakIndex-1;i++)
        {
            PeakFreq[i]=PeakFreq[i+1];
        }
        pcount=pcount-1;
    }

    //calculate vpd
    for(int i=0;i<pcount;i++)
    {
        vpd[i]=Peaks[i]-valleys[i];
    }

    //peaks screening
    int delcount;
    int dels[256]{0};
    int peakF[256]{0};
    float vpd1[256]{0};
    float freqs[256]{0};
    float ampPeaks[256]{0};

    if(pcount>2)
    {
        int lastcount=pcount;
        int curcount=0;
        while(lastcount!=curcount)
        {
            lastcount=curcount;
            delcount=0;
            for(int i=1;i<pcount-1;i++)
            {
                if(vpd[i]<=0.5*(vpd[i-1]+vpd[i]+vpd[i+1])/3.0f)
                {
                    dels[i]=1;
                }
            }

            int count=0;
            for(int i=0;i<pcount;i++)
            {
                if(dels[i]!=1)
                {
                    peakF[count]=PeakFreq[i];
                    vpd1[count]=vpd[i];
                    count=count+1;
                }
                else
                {
                    delcount=delcount+1;
                    dels[i]=0;
                }
            }

            pcount=pcount-delcount;
            for(int i=0;i<pcount;i++)
            {
                PeakFreq[i]= peakF[i];
                vpd[i]=vpd1[i];
            }

            PeakFreq[pcount+1]=0;
            vpd[pcount+1]=0;

            curcount=pcount;
        }
    }

    for(int i=0;i<pcount;i++)
    {
        freqs[i]=static_cast<float>(PeakFreq[i]*m_sampfreq/1024) ;
        ampPeaks[i]=channelData[PeakFreq[i]];
    }
    quickSort(0,pcount-1,ampPeaks,freqs);

    //source number control,take the top ten
    SourceInf *PeakandCount =new SourceInf;
    PeakandCount->SourceNums=0;
    int i=0,j=0;
    float MinDetectPeak=3.0f;
    while(i<MaxSourceNum)
    {
        if(freqs[i]>MinDetectFreq && ampPeaks[i]>MinDetectPeak)
        {
            PeakandCount->SourceNums++;
            PeakandCount->SourceFreq[j]=freqs[i];
            j++;
        }
        i++;
    }
    return  PeakandCount;
}
void quickSort(int left,int right,float data[],float index[])
{
    if(left >= right)
        return;
    int i, j;
    float base, base1,temp;
    i = left;
    j = right;
    base=data[left];
    base1=index[left];
    while(i<j)
    {
        while (data[j] <= base && i < j)
            j--;
        while (data[i] >= base && i < j)
            i++;
        if(i < j)
        {
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            temp =index[i];
            index[i]=index[j];
            index[j]=temp;
        }
    }
    data[left] = data[i];
    index[left]=index[i];
    data[i] = base;
    index[i] =base1;
    quickSort(left, i - 1, data,index);//递归左边
    quickSort(i + 1, right,data,index);//递归右边
}

__global__ void zeros(float* input)
{
    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    input[threadID]=0.0f;
}
__host__ float max(float* vector, int size)
{
    float max_value = 0;
    for (int i = 0; i < size; i++)
    {
        if (vector[i] - max_value > 0)
            max_value = vector[i];
    }
    return max_value;
}
__host__  float min(float* vector, int size)
{
    float min_value = vector[0];
    for (int i = 0; i < size; i++)
    {
        if (vector[i] < min_value)
            min_value = vector[i];
    }
    return  min_value;
}
