/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#define DATATYPE double
#define VL 2
#define VECTYPE double2

#include <cstdio>
#include "device_reduce_atomic.h"
#include "device_reduce_block_atomic.h"
#include "device_reduce_warp_atomic.h"
#include "device_reduce_stable.h"
#include "vector_functions.h"
#include "cub/cub.cuh"
//#include "cub/cub/cub.cuh"

#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                  \
  if(e!=cudaSuccess) {                                               \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
  exit(0); \
  }                                                                  \
}

template <typename T>
void RunTest(const char* label, void (*fptr)(T* in, T* out, int N), int N, int REPEAT, T* src, T checksum) {
  T *in, *out;
  
  //allocate a buffer that is at least large enough that we can ensure it doesn't just sit in l2.
  //int MIN_SIZE=4*1024*1024;
  //int size=max(int(sizeof(T)*N),MIN_SIZE);
  
  //compute mod base for picking the correct buffer
  //int mod=size/(N*sizeof(T));
  cudaEvent_t start,stop;
  //cudaMalloc(&in,size);
  cudaMalloc(&in,sizeof(T)*N);
  cudaMalloc(&out,sizeof(T)*1024);  //only stable version needs multiple elements, all others only need 1
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaCheckError();

  cudaMemcpy(in,src,N*sizeof(T),cudaMemcpyHostToDevice);
  
  //warm up
  //fptr(in,out,N);

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaEventRecord(start);

  for(int i=0;i<REPEAT;i++) {
    //iterate through different buffers
    //int o=i%mod;
    //fptr(in+o*N,out,N);
    fptr(in,out,N);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaCheckError();

  float time_ms;
  cudaEventElapsedTime(&time_ms,start,stop);
  float time_s=time_ms/(float)1e3;

  float GB=(float)N*sizeof(T)*REPEAT;
  float GBs=GB/time_s/(float)1e9;

  T sum;
  cudaMemcpy(&sum,out,sizeof(T),cudaMemcpyDeviceToHost);
  cudaCheckError();

  static const char *valid;
  if(sum==checksum)
    valid="CORRECT";
  else
    valid="INCORRECT";

  printf("%s: \033[31m%s\033[0m, Time: %f s, GB/s: \033[32m%f\033[0m\n", label, valid, time_s, GBs); 
  if(sum!=checksum) printf("sum=%f checksum=%f diff=\033[33m%f\033[0m\n\n", sum, checksum, checksum-sum);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(in);
  cudaFree(out);
  cudaCheckError();
}

template <typename T>
void RunTestCub(const char* label, int N, int REPEAT, T* src, T checksum) {
  T *in, *out;
  cudaEvent_t start,stop;
  
  cudaMalloc(&in,sizeof(T)*N);
  cudaMalloc(&out,sizeof(T)*1024);  //only stable version needs multiple elements, all others only need 1
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaCheckError();

  cudaMemcpy(in,src,N*sizeof(T),cudaMemcpyHostToDevice);

  size_t temp_storage_bytes;
  T* temp_storage=NULL;
  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, in, out, N);
  cudaMalloc(&temp_storage,temp_storage_bytes);

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaEventRecord(start);

  for(int i=0;i<REPEAT;i++) {
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, in, out, N);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaCheckError();

  float time_ms;
  cudaEventElapsedTime(&time_ms,start,stop);
  float time_s=time_ms/(float)1e3;

  float GB=(float)N*sizeof(T)*REPEAT;
  float GBs=GB/time_s/(float)1e9;

  T sum;
  cudaMemcpy(&sum,out,sizeof(T),cudaMemcpyDeviceToHost);
  cudaCheckError();

  static const char *valid;
  if(sum==checksum)
    valid="CORRECT";
  else
    valid="INCORRECT";

  printf("%s: \033[31m%s\033[0m, Time: %f s, GB/s: \033[32m%f\033[0m\n", label, valid, time_s, GBs);
  if(sum!=checksum) printf("sum=%d checksum=%f diff=\033[33m%d\033[0m\n\n", sum, checksum, checksum-sum);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(in);
  cudaFree(out);
  cudaFree(temp_storage);
  cudaCheckError();
}

int main(int argc, char** argv)
{
  if(argc!=3) {
    printf("Usage: ./reduce num_elems repeat\n");
    exit(0);
  }
  int NUM_ELEMS=atoi(argv[1]);
  int REPEAT=atoi(argv[2]);

  printf("NUM_ELEMS: %d, REPEAT: %d, DATATYPE: %s\n", NUM_ELEMS, REPEAT, typeid(DATATYPE).name());

  DATATYPE* vals=(DATATYPE*)malloc(NUM_ELEMS*sizeof(DATATYPE));
  int csum = 0;
  for(int i=0;i<NUM_ELEMS;i++) {
    //vals[i]=(DATATYPE)(rand()%4);
    vals[i]= (DATATYPE)(2-rand()%5);
    csum+=(int)vals[i];
    //printf("Main: %d %d %d \n",i,vals[i],csum);
  }
  DATATYPE checksum = (DATATYPE)csum;
  //printf("Main: checksum=%d \n",checksum);

  RunTest<DATATYPE>("device_reduce_atomic", device_reduce_atomic,NUM_ELEMS,REPEAT,vals,checksum);
  RunTest<DATATYPE>("device_reduce_atomic_vector", device_reduce_atomic_vector<DATATYPE,VECTYPE,VL>,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_atomic_vector2", device_reduce_atomic_vector2,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_atomic_vector4", device_reduce_atomic_vector4,NUM_ELEMS,REPEAT,vals,checksum);
  
  RunTest<DATATYPE>("device_reduce_warp_atomic",device_reduce_warp_atomic,NUM_ELEMS,REPEAT,vals,checksum);
  RunTest<DATATYPE>("device_reduce_warp_atomic_vector",device_reduce_warp_atomic_vector<DATATYPE,VECTYPE,VL>,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_warp_atomic_vector2",device_reduce_warp_atomic_vector2,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_warp_atomic_vector4",device_reduce_warp_atomic_vector4,NUM_ELEMS,REPEAT,vals,checksum);
  
  RunTest<DATATYPE>("device_reduce_block_atomic",device_reduce_block_atomic,NUM_ELEMS,REPEAT,vals,checksum);
  RunTest<DATATYPE>("device_reduce_block_atomic_vector",device_reduce_block_atomic_vector<DATATYPE,VECTYPE,VL>,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_block_atomic_vector2",device_reduce_block_atomic_vector2,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_block_atomic_vector4",device_reduce_block_atomic_vector4,NUM_ELEMS,REPEAT,vals,checksum);
  
  RunTest<DATATYPE>("device_reduce_stable",device_reduce_stable,NUM_ELEMS,REPEAT,vals,checksum);
  RunTest<DATATYPE>("device_reduce_stable_vector",device_reduce_stable_vector<DATATYPE,VECTYPE,VL>,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_stable_vector2",device_reduce_stable_vector2,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_stable_vector4",device_reduce_stable_vector4,NUM_ELEMS,REPEAT,vals,checksum);

  RunTestCub<DATATYPE>("device_reduce_cub",NUM_ELEMS,REPEAT,vals,checksum);
  
  free(vals);

}
