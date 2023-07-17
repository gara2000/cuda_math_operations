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

const int TILE_SIZE = 16;
const int SHARED_MEM_SIZE = 16 * 16;

__global__ void cache_tiled_mat_mul(int *res, int* a, int* b, int N)
{
	// Two statically-sized pieces of shared memory
	__shared__ int A[SHARED_MEM_SIZE];
	__shared__ int B[SHARED_MEM_SIZE];


	int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;

	A[ty * TILE_SIZE + tx] = 0;
	B[ty * TILE_SIZE + tx] = 0;
	__syncthreads();

	// Global row and global column
	int row = ty + blockDim.y * by;
	int col = tx + blockDim.x * bx;

	if(row<N&&col<N)
	{

		// Sweep tiles over the entire matrix
		int tmp = 0;
		for(int i=0; i<(N+TILE_SIZE-1)/TILE_SIZE; i++)
		{
		/*
		   *Every thread in a threablock loads one element into shared memory.
		   *The element location in shared memory corresponds to the thread's location in the threadblock.

		   *Indexing parameters:
			For shared mem A:
				row : the global row
				TILE_SIZE * i : set of columns to choose from for each iteration
				tx : the column in this set
				
			For shared mem B:
				TILE_SIZE * i * N : set of rows to choose from for each iteration
				ty * N : the row in this set
				col : the global column
		*/

			if(row < N && (TILE_SIZE * i + tx) < N)
				A[(TILE_SIZE * ty) + tx] = a[row*N + (TILE_SIZE * i + tx)];

			if(TILE_SIZE * i + ty < N && col < N)
				B[(TILE_SIZE * ty) + tx] = b[(TILE_SIZE * i * N + ty * N) + col];

			// Synchronize the threads to ensure that all the threads have loaded their data in the shared memory
			__syncthreads();

			//add to the corresponding cell in the result matrix
			for(int j = 0; j<TILE_SIZE; j++)
				tmp += A[(TILE_SIZE * ty) + j] * B[(TILE_SIZE * j) + tx];

			// Ensure that no thread progress and change current shared memory values before other threads have read from it
			__syncthreads();
		}
		res[(row * N) + col] = tmp;
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
	const int N=1<<8;
	int bytes = N*N*sizeof(int);

	int *a = (int*)malloc(bytes);
	int *b = (int*)malloc(bytes);
	int *res = (int*)malloc(bytes);

	fill_vec(a, N); fill_vec(b, N);

	//repr(a, N); repr(b, N);	

	int *d_a, *d_b, *d_res;
	
	cudaMalloc((void**)&d_a, bytes); 
	cudaMalloc((void**)&d_b, bytes);
	cudaMalloc((void**)&d_res, bytes);

	//Recording the time taken by the kernel
	auto start = std::chrono::high_resolution_clock::now();

	cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

	// the sizes are set such that each row in the grid represents a row in the matrix
	// GRID_SIZE * BLOCK_SIZE = N
	int BLOCK_SIZE = 16;
	int GRID_SIZE = (N+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 blk(BLOCK_SIZE, BLOCK_SIZE);

	cache_tiled_mat_mul<<<grid, blk>>>(d_res, d_a, d_b, N);

	cudaMemcpy(res, d_res, bytes, cudaMemcpyDeviceToHost);
	
	auto stop = std::chrono::high_resolution_clock::now();

	auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
	cout<<"Elapsed time: "<<(float)time.count()/1000000<<" ms"<<endl;

	verify(res, a, b, N);

	cudaFree(d_a); cudaFree(d_b);cudaFree(d_res);
	
	return 0;
}
