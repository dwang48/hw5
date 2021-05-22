#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#define SIZE_BLOCK 1024

__global__ void reduction(double* new_u, double* f, double* u, long N, double h){
  __shared__ double temp[SIZE_BLOCK];
  
  int index = (blockIdx.x)*blockDim.x+threadIdx.x;
  if(index >= N && index < N*(N-1) && index % N > 0 && index % N < N - 1 ){
    double left = u[index - 1];    
    double right = u[index + 1];   
    double bottom = u[index + N];    
    double top = u[index - N];   

    temp[threadIdx.x] = (left + right + top + bottom + h*h*f[index]) * 0.25;

  }else temp[threadIdx.x] = 0;

    __syncthreads();
    if (threadIdx.x<32) {
      temp[threadIdx.x] += temp[threadIdx.x + 32];
      __syncwarp();
      temp[threadIdx.x] += temp[threadIdx.x + 16];
      __syncwarp();
      temp[threadIdx.x] += temp[threadIdx.x +  8];
      __syncwarp();
      temp[threadIdx.x] += temp[threadIdx.x +  4];
      __syncwarp();
      temp[threadIdx.x] += temp[threadIdx.x +  2];
      __syncwarp();
      if (threadIdx.x == 0) new_u[blockIdx.x] = temp[0] + temp[1];
    }

    __syncthreads();

    if (threadIdx.x<64) temp[threadIdx.x] += temp[threadIdx.x + 64];

    __syncthreads();

    if (threadIdx.x<128) temp[threadIdx.x] += temp[threadIdx.x + 128];

    __syncthreads();

    if (threadIdx.x<256) temp[threadIdx.x] += temp[threadIdx.x + 256];

     __syncthreads();

    if (threadIdx.x<512) temp[threadIdx.x] += temp[threadIdx.x + 512];
    
    
}

//CUDA Version Jacobi
__global__ void jacobi_2d(double* new_u, double* f, double* u, long N, double h)
{

    int index = (blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= N && index < N*(N-1) && index % N > 0 && index % N < N - 1){
          double left = u[index - 1];    
          double right = u[index + 1];   
          double bottom = u[index + N];    
          double top = u[index - N];       

        new_u[index] = (left + right + top + bottom + h*h*f[index]) * 0.25;
    }
}

void VecSum(double* ptr, const double* a, long N){
    double sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (long i = 0; i < N*N; i++) sum += a[i];
    *ptr = sum;
}

void normal_jacobi_2d(double* new_u, double* f, double* u, long N, double h){
    
    for (long i = 1; i < N; i++){
        
        for (long j = i*N; j < (i+1)*N-1; j++){

          if(j % N > 0 && j % N < N - 1 ){
              double left = u[j - 1];    
              double right = u[j + 1];   
              double top = u[j - N];    
              double bottom = u[j + N];    
              new_u[j] = (h*h*f[j] + left + right + top + bottom ) * 0.25;   
          }
        }
    }
}

void CheckError(const char *message){
  cudaError_t ERROR = cudaGetLastError();
  if(ERROR!=cudaSuccess){
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(ERROR) );
    exit(-1);
  }
}




int main() {
  long N = 100;
  /*Initialization*/
  double h = 1.0/(N+1.0), sum = 0.0, correct_sum = 0.0;
  double *u,*new_u,*temp_u,*f;

  cudaMallocHost((void**)&u, N * N * sizeof(double));
  cudaMallocHost((void**)&new_u, N * N * sizeof(double));
  cudaMallocHost((void**)&temp_u, N * N * sizeof(double));
  cudaMallocHost((void**)&f, N * N * sizeof(double));


  memset(u,0,N*N);
  memset(new_u,0,N*N);
  memset(temp_u,0,N*N);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N*N; i++) f[i] = 1.0;

  /*Normal Jacobi Computation*/
  for (long k = 0; k < 80; k++){
      normal_jacobi_2d( new_u, f, u, N, h);
      for (long i = 1; i < N*N; i++)
          u[i] = new_u[i];
  }
  VecSum(&correct_sum, new_u, N);

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N*N; i++) new_u[i] = 0.0;
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N*N; i++) u[i] = 0.0;

  double tt = omp_get_wtime();
  double *temp_u_c, *u_c, *f_c;
  cudaMalloc(&temp_u_c, N*N*sizeof(double));
  cudaMalloc(&u_c, N*N*sizeof(double));
  cudaMalloc(&f_c, N*N*sizeof(double));
 

  cudaMemcpyAsync(temp_u_c, temp_u, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(u_c, u, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaMemcpyAsync(f_c, f, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  tt = omp_get_wtime();

  /*CUDA Version Jacobi Computation*/
  for (long k = 0; k < 80; k++){
      jacobi_2d <<< N, SIZE_BLOCK >>> (temp_u_c, f_c, u_c, N, h);
      u_c = temp_u_c;
  }
  cudaMemcpyAsync(temp_u, temp_u_c, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  VecSum(&sum, temp_u, N);
  printf("Absolute Error= %f\n", fabs(sum-correct_sum));
  printf("Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("correct_sum: %f\n", correct_sum);
  printf("sum: %f\n", sum);

  cudaFree(temp_u_c);
  cudaFree(u_c);
  cudaFree(f_c);
  cudaFreeHost(f);
  cudaFreeHost(temp_u);
  cudaFreeHost(new_u);
  cudaFreeHost(u);
  return 0;
}



