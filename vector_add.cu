#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;

template<typename T>
void repr(T* a, int N)
{
	for(int i=0;i<N;i++)
		printf("%d ", a[i]);
	printf("\n");
}
template<typename T>
void fill_vec(T* a, int N)
{
	for(int i=0;i<N;i++)
		a[i] = random()%100;
}

const int SIZE = 32;

__global__ void add_vect_gpu(int *res, int *a, int*b, int N)
{
	__shared__ int A[SIZE];
	__shared__ int B[SIZE];
	
	int tx = threadIdx.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	A[tx] = a[i];
	B[tx] = b[i];

	if(i<N)
		res[i] = A[tx]+B[tx];
}

bool verify(int *res, int *a, int *b, int N)
{
	for(int i=0;i<N;i++)
		if(res[i]!=a[i]+b[i])
			return false;
	return true;
}

int main(void)
{
	// vectors length
	int N = 100000;

	int size = N*sizeof(int);
	int *a=(int*)malloc(size);
	int *b=(int*)malloc(size);
	int *res=(int*)malloc(size);

	// Initialize the vectors a and b
	fill_vec(a, N); fill_vec(b, N);

	//repr(a, N); repr(b, N);	

	int *d_a, *d_b, *d_res;
	
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_res, size);

//Recording the time taken by the kernel
auto start = std::chrono::high_resolution_clock::now();

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	const int BLK_SIZE = SIZE;
	const int GRID_SIZE = (N+BLK_SIZE-1)/BLK_SIZE;

	// Call the kernel
	add_vect_gpu<<<GRID_SIZE, BLK_SIZE>>>(d_res, d_a, d_b, N);

	cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);

auto stop = std::chrono::high_resolution_clock::now();

auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
cout<<"Elapsed time: "<<(float)time.count()/1000000<<" ms"<<endl;

	// Verify the results
	if(verify(res, a, b, N))
		cout<<"Correct results!"<<endl;
	else
		cout<<"Wrong results!"<<endl;

	cudaFree(d_a); cudaFree(d_b);cudaFree(d_res);
	
	return 0;
}
