# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-10.2

HOST_COMPILER ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
TARGET = Beamforming

#internal flags
sm?=53
GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)
# NVCCFLAGS += -m64 -g -G -use_fast_math
NVCCFLAGS +=-m64 -O3 -use_fast_math

CCFLAGS += $(NVCCFLAGS) $(GENCODE_FLAGS)
LDFLAGS += $(CCFLAGS)

INCLUDES :=-I $(CUDA_PATH)/samples/common/inc -I $(CUDA_PATH)/include
LIBRARIES +=-lcufft -lcublas -lcudart -L $(CUDA_PATH)/lib64

CPP_SOURCE=$(wildcard *.cpp)
CUDA_SOURCE=$(wildcard *.cu)

CPPOBJS=$(CPP_SOURCE:.cpp=.o )
CUOBJS=$(CUDA_SOURCE:.cu=.o )
OBJS = $(CUOBJS) $(CPPOBJS)

all:$(TARGET)

%.o: %.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(LDFLAGS) -o $@ -c $<
$(CPPOBJS):$(CPP_SOURCE)
	$(EXEC) $(HOST_COMPILER) -Wall $(INCLUDES) -o $@ -c $<
$(TARGET):$(OBJS)
	$(EXEC) $(NVCC) $(LDFLAGS) -o $@ $+ $(LIBRARIES)

.PHONY:clean
clean:
	-rm -rf $(TARGET)
	-rm -rf $(OBJS)

