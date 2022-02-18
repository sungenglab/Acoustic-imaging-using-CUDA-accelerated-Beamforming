#include "SourceFinding.h"
#include <iostream>
#include <unordered_map>

using namespace std;

__device__ float static_oridata[ARRAYNUM * len];
__device__ cufftComplex static_complex[ARRAYNUM * len];
__device__ cufftComplex static_amplitude[ARRAYNUM * len];
__device__ cufftComplex static_Channel[ARRAYNUM * len];
__device__ cufftComplex static_maxmagfreq[ARRAYNUM * len];

SourceFinding::SourceFinding():
    m_sampfreq(70000.0f),
    MinDetectFreq(2000.0f),
    MinDetectPeak(3.0f),
    ThreadsPerBlock_0(32, 32),
    BlocksPerGrid_0(ARRAYNUM / ThreadsPerBlock_0.x, len / ThreadsPerBlock_0.y),
    th(1, 1024),
    b(32, 1)
{
    SourceDetect = new Sources();
    int BATCH = ARRAYNUM;
    int rank = 1;
    int n1[1] = { len };
    int istride = ARRAYNUM;
    int idist = 1;
    int ostride = ARRAYNUM;
    int odist = 1;
    int inembed[2] = { ARRAYNUM, len };
    int onembed[2] = { ARRAYNUM, len };

    checkCudaErrors(cufftPlanMany(&PlanManyC2C, rank, n1, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, BATCH));
    checkCudaErrors(cudaMallocHost((void**)&H_channelData, len * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&H_maxmagfreq, ARRAYNUM * len * sizeof(float)));
}
SourceFinding::~SourceFinding()
{
    checkCudaErrors(cufftDestroy(PlanManyC2C));
    checkCudaErrors(cudaFreeHost(H_channelData));
    checkCudaErrors(cudaFreeHost(H_maxmagfreq));
}
Sources* SourceFinding::PeaksFinding(float* channelData,float sampFreq)
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
        freqs[i]=static_cast<float>(PeakFreq[i]*sampFreq/1024) ;
        ampPeaks[i]=channelData[PeakFreq[i]];
    }

    quickSort(0,pcount-1,ampPeaks,freqs);

    //source number control,take the top ten
    Sources *PeakandCount =new Sources;
    int i=0,j=0;
    while(i<MaxNum)
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
void SourceFinding::quickSort(int left,int right,float data[],float index[])
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

void SourceFinding::finding()
{
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_float, static_oridata));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_complex, static_complex));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_amplitude, static_amplitude));
    checkCudaErrors(cudaGetSymbolAddress((void**)&channelData, static_Channel));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_maxmagfreq, static_maxmagfreq));

    checkCudaErrors(cudaMemcpy(d_float, m_oridata,
                               sizeof(float) * ARRAYNUM * len,
                               cudaMemcpyHostToDevice));

    Cvt2Complex << < BlocksPerGrid_0, ThreadsPerBlock_0 >> > (d_float, d_complex);
    cudaError_t error0 = cudaGetLastError();
    if (!(error0 == cudaSuccess))
        printf("kernel Real2Complex: %s\n", cudaGetErrorString(error0));

    checkCudaErrors(cufftExecC2C(PlanManyC2C, d_complex, d_complex, CUFFT_FORWARD));

    getamplitude << <BlocksPerGrid_0, ThreadsPerBlock_0>> > (d_complex, d_amplitude);
    cudaError_t error1 = cudaGetLastError();
    if (!(error1 == cudaSuccess))
        printf("kernel getSignal_amplitude: %s\n", cudaGetErrorString(error1));

    getPeakFreqs << <b, th >> > (d_amplitude, m_sampfreq, d_maxmagfreq);
    cudaError_t error3 = cudaGetLastError();
    if (!(error3 == cudaSuccess))
        printf("kernel getPeakFreq: %s\n", cudaGetErrorString(error3));

    checkCudaErrors(cudaMemcpy(H_maxmagfreq, d_maxmagfreq, ARRAYNUM * sizeof(float), cudaMemcpyDeviceToHost));

    UsedChannnel = Top1Frequent(H_maxmagfreq,ARRAYNUM);

    ChannelSelect<<<b, th>>>(d_amplitude,channelData,UsedChannnel->channelID);

    checkCudaErrors(cudaMemcpy(H_channelData, channelData, len * sizeof(float), cudaMemcpyDeviceToHost));

    SourceDetect=PeaksFinding(H_channelData,m_sampfreq);
    cout<<"Source Detected :"<<SourceDetect->SourceNums<<endl;
    for(int i=0;i<SourceDetect->SourceNums;i++)
    {
        cout<<"source"<<i<<": "<<SourceDetect->SourceFreq[i]<<" Hz"<<endl;
    }
}

__global__ void Cvt2Complex(float* matrix_in, cufftComplex* matrix_out)
{
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadID < ARRAYNUM * len)
    {
        matrix_out[threadID].x = matrix_in[threadID];
        matrix_out[threadID].y = 0.0f;
    }
}
__global__ void getamplitude(cufftComplex* fftresult, float* sign_mag)
{
    unsigned int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadID < 1024 * 32)
    {
        sign_mag[threadID] = sqrtf(fftresult[threadID].x * fftresult[threadID].x + fftresult[threadID].y * fftresult[threadID].y);
    }
}
__global__ void ChannelSelect(float* Amps,float *channelData,int ChannelID)
{
    unsigned int i = threadIdx.y;
    channelData[i]=0.0f;
    channelData[i]=Amps[i * 32+ChannelID];

}
Channels* SourceFinding::Top1Frequent(float* vector, int size)
{
    Channels *max =new Channels;
    max->PeakFreq=vector[0];
    max->channelID=0;
    //Channel with the highest value
    if(0)
    {
        for (int i = 0; i < size; i++)
        {
            if (vector[i] - max->PeakFreq > 0)
            {
                max->PeakFreq = vector[i];
                max->channelID =i;
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
        it++;
    }
    max->PeakFreq=vector[channelID];
    max->channelID=static_cast<int>(index[channelID]);

    return max;
}
__global__ void getPeakFreqs(float* signal_mag, float freq_samp, float* freq_mag)
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
