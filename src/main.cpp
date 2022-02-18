#include <iostream>
#include "fstream"

#include "FIRFilter.h"
#include "SoundFieldCal.h"

int main(int argc, char* argv[])
{
    if(argc!=5){
        std::cout<<"Help: this program at least needs 5 params"<<std::endl;
        std::cout<<"1: Array Position path"<<std::endl;
        std::cout<<"2: Data path"<<std::endl;
        std::cout<<"3: Sample Frequency"<<std::endl;

    }
    
    float* input_data=nullptr;
    float* sound_matrix=nullptr;
    float  Peakfreq =0.0f;
    float  centralSPL =0.0f;

    input_data = (float*)malloc(sizeof(float)*(1024 * 32));

    ifstream data;
    
    data.open(argv[1],ios::in);
    if (!data.is_open())
    {
        cout << "读取数据文件失败" << endl;
        return;
    }
    string buf;
    int i=0;
    while (getline(data, buf))
    {
        input_data[i]=stof(buf);
        i++;
    }

    SoundFieldCal *GPUCal=new SoundFieldCal;
    
    GPUCal->setData(input_data);
    GPUCal->setImageThre(0.6f);
    GPUCal->calculate();
    Peakfreq=GPUCal->getMaxfreq();
    centralSPL=GPUCal->getSPL();
    sound_matrix=GPUCal->getSoundResult();

    cout<<"result Peakfreq:"<<Peakfreq<<endl;
    cout<<"result CentralSPL:"<<centralSPL<<endl;

    return 0;
}