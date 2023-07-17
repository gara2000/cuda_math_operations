#include <stdio.h> 
#include <chrono>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

template<typename T>
void repr(T *a, int N)
{
	for(int i=0;i<N;i++)
		printf("%d ", a[i]);
	printf("\n");
}

void init_vector(int* a, int N)
{
	for(int i=0;i<N;i++)
		a[i] = rand()%10;
}

const int SIZE = 32; 
const int SHMEM_SIZE = SIZE;
const int Na = 1<<27;
const int Nb = 16; //Nb should be smaller or equal to SHMEM_SIZE

__global__ void conv1D(int *res, int*a, int*b, int Na, int Nb)
{
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int i = tid + bid * blockDim.x;
	if(i<Na)
	{
		A[tid] = a[i];
		if(tid<Nb)
			B[tid] = b[tid];

		res[i] = 0;
		__syncthreads();

		for(int j = 0; j<Nb; j++)
		{
			if(i - j >= 0)
				atomicAdd(&res[i-j], A[tid]*B[j]);
			__syncthreads();
		}
	}
}

void cpu_conv(int *res, int *a, int *b, int Na, int Nb)
{
	for(int i=0;i<Na; i++)
		res[i] = 0;
	for(int i=0;i<Na;i++)
	{
		for(int j=0;j<Nb;j++)
		{
			if(i-j>=0)
				res[i-j] += a[i] * b[j];
		}
	}
}

void verify(int *d_res, int *h_res) 
{
	for(int i=0;i<Na-Nb+1;i++)
		if(d_res[i]!=h_res[i])
		{
			printf("Wrong result on index %d, expected %d but found %d !\n", i, h_res[i], d_res[i]);
			return;
		}
	printf("Correct result!\n");
}

int main()
{
	srand(0);

	int size_a = Na * sizeof(int);
	int size_b = Nb * sizeof(int);

	int *a, *d_a, *b, *d_b;
	int *res, *d_res, *cpu_res;

	a = (int*)malloc(size_a);
	b = (int*)malloc(size_b);
	res = (int*)malloc(size_a);
	cpu_res = (int*)malloc(size_a);
	init_vector(a, Na);
	init_vector(b, Nb);

	cudaMalloc((void**)&d_a, size_a);
	cudaMalloc((void**)&d_b, size_b);
	cudaMalloc((void**)&d_res, size_a);

	{// convolution on gpu
		auto start = std::chrono::high_resolution_clock::now();
		cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice); 
		cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice); 
		cudaMemcpy(d_res, res, size_a, cudaMemcpyHostToDevice);

		int BLOCK_SIZE = SIZE;
		int GRID_SIZE = (Na+BLOCK_SIZE-1)/BLOCK_SIZE;
		printf("Grid size: %d, Block size: %d\n", GRID_SIZE, BLOCK_SIZE) ;

		conv1D<<<GRID_SIZE, BLOCK_SIZE>>>(d_res, d_a, d_b, Na, Nb);


		cudaMemcpy(res, d_res, size_a, cudaMemcpyDeviceToHost);

		auto stop = std::chrono::high_resolution_clock::now();

		auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
		cout<<"Elapsed time on GPU: "<<(float)time.count()/1000000<<" ms"<<endl;
	}
	{// convolution on cpu
		auto start = std::chrono::high_resolution_clock::now();

		cpu_conv(cpu_res, a, b, Na, Nb);

		auto stop = std::chrono::high_resolution_clock::now();

		auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
		cout<<"Elapsed time on CPU: "<<(float)time.count()/1000000<<" ms"<<endl;
	}

	verify(res, cpu_res);

	return 0;
}
