#include <stdio.h> 
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

void init_vector(int* a, int N)
{
	for(int i=0;i<N;i++)
		a[i] = rand()%100;
}

const int SIZE = 128;
const int SHMEM_SIZE = SIZE;
__global__ void sum_reduction(int* res, int* a, int N)
{
	int tx = threadIdx.x;
	__shared__ int A[SHMEM_SIZE];

	int tid = tx + blockDim.x * blockIdx.x;
	if(tid<N)
		A[tx] = a[tid];
	else
		A[tx] = 0;
	__syncthreads();

	for(int s=blockDim.x/2; s>0; s>>=1)
	{
		if(tx<s)
			atomicAdd(&A[tx], A[tx+s]);
		__syncthreads();
	}

	if(tx==0)
		res[blockIdx.x] = A[0];
}
			
bool verify(int *res, int *a, int N)
{
	int sum = 0;
	for(int i=0;i<N;i++)
		sum+=a[i];
	if(sum==*res)
		return true;
	return false;
}

int main()
{
	int N = 1000000;
	int tmp = N;
	int size = N * sizeof(int);

	int *a, *d_a;
	int *res, *d_res;

	a = (int*)malloc(size);
	res = (int*)malloc(size);
	init_vector(a, N);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_res, size);

	auto start = std::chrono::high_resolution_clock::now();

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_res, res, sizeof(int), cudaMemcpyHostToDevice);

	int BLOCK_SIZE = SIZE;
	int GRID_SIZE = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
//	printf("blk: %d, grid: %d\n", BLOCK_SIZE, GRID_SIZE);

	if(GRID_SIZE!=1)
	{
		sum_reduction<<<GRID_SIZE, BLOCK_SIZE>>>(d_res, d_a, N);
		N = GRID_SIZE;
		GRID_SIZE = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
	}
	while(GRID_SIZE>1)
	{
//		printf("blk: %d, grid: %d\n", BLOCK_SIZE, GRID_SIZE);
		sum_reduction<<<GRID_SIZE, BLOCK_SIZE>>>(d_res, d_res, N);
		N = GRID_SIZE;
		GRID_SIZE = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
	}

	sum_reduction<<<1, BLOCK_SIZE>>>(d_res, d_res, N);

	cudaMemcpy(res, d_res, sizeof(int), cudaMemcpyDeviceToHost);

	auto stop = std::chrono::high_resolution_clock::now();

	auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
	cout<<"Elapsed time on GPU: "<<(float)time.count()/1000000<<" ms"<<endl;

	N = tmp;
	if(verify(res, a, N))
		printf("Correct Result!\n");
	else
		printf("Wrong Result!\n");

	return 0;
}
