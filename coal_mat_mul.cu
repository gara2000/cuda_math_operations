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
	for(int i=0;i<N*N;i++)
	{
		printf("%d ", a[i]);
		if((i+1)%N==0)
			printf("\n");
	}
	printf("\n");
}
template<typename T>
void fill_vec(T* a, int N)
{
	for(int i=0;i<N*N;i++)
		a[i] = rand()%10;
}

__global__ void mat_mul_coal(int *res, int *a, int *b, int N)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row>=N || col>=N)
		return;

	res[row * N + col] = 0;
	for(int k = 0; k<N; k++)
	{
		res[row * N + col] += a[row + k * N] * b[col + k * N];
	}
}

void mat_mul_cpu(int *res, int *a, int *b, int N)
{
	for(int row=0;row<N;row++)
		for(int col=0;col<N;col++)
		{
			res[row * N + col] = 0;
			for(int k=0;k<N;k++)
				res[row*N+col] += a[row * N + k] * b[col + k * N];
		}
}	


void verify(int *res, int* a, int* b, int N)
{
	int ver[N*N];
	mat_mul_cpu(ver, a, b, N);
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++)
			if(ver[i*N+j]!=res[i*N+j])
			{
				cout<<"Wrong result!"<<endl;
				return;
			}
	cout<<"Correct Result!"<<endl;
	return ;
}

void transpose(int* a_t, int* a, int N)	
{
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++)
			a_t[i*N+j] = a[j*N+i];
}

int main(void)
{
	const int N=100;
	int bytes = N*N*sizeof(int);

	int *a = (int*)malloc(bytes);
	int *b = (int*)malloc(bytes);
	int *res = (int*)malloc(bytes);
	int *a_t = (int*)malloc(bytes);

	fill_vec(a, N); fill_vec(b, N);
	transpose(a_t, a, N);

	//repr(a, N); repr(b, N);	

	int *d_a, *d_b, *d_res, *d_a_t;
	
	cudaMalloc((void**)&d_a, bytes); 
	cudaMalloc((void**)&d_b, bytes);
	cudaMalloc((void**)&d_res, bytes);
	cudaMalloc((void**)&d_a_t, bytes);

	//Recording the time taken by the kernel
	auto start = std::chrono::high_resolution_clock::now();

	cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a_t, a_t, bytes, cudaMemcpyHostToDevice);

	// the sizes are set such that each row in the grid represents a row in the matrix
	// GRID_SIZE * BLOCK_SIZE = N
	int BLOCK_SIZE = 16;
	int GRID_SIZE = (N+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 blk(BLOCK_SIZE, BLOCK_SIZE);

	mat_mul_coal<<<grid, blk>>>(d_res, d_a_t, d_b, N);

	cudaMemcpy(res, d_res, bytes, cudaMemcpyDeviceToHost);
	
	auto stop = std::chrono::high_resolution_clock::now();

	auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
	cout<<"Elapsed time: "<<(float)time.count()/1000000<<" ms"<<endl;

	verify(res, a, b, N);

	cudaFree(d_a); cudaFree(d_b);cudaFree(d_res);
	
	return 0;
}
