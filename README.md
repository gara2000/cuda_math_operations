# PLATFORM TESTING, README

##Description:
This folder contains 4 different cuda kernels:
	- Vector Addition kernel: vector_add.cu
		This kernel performs simple vector addition operation of two vectors a and b of size N

	- 3 different Matrix Multiplication kernels: naive_mat_mul.cu, coal_mat_mul.cu and cache_tiled_mat_mul.cu
		Each of these three kernels performs the matrix multiplication operation of two matrices a and b of size N*N (randomly initialized)

	- 1D Convolution kernel: conv.cu
		This kernel perform the convolution operation of a 1D vector of size N by a 1D filter of small size

	- Vector Sum Reduction kernel: sum_red.cu
		This kernel calculates the sum of elements of a vector a of size N

## Used system packages
	nvidia-cuda-toolkit	10.1.243
	git	2.25.1 

## Cloning the repository:
	$ git clone https://github.com/gara2000/cuda_math_operations.git 

## Building the project: this will create a compiled code file for each cuda file
	$ make
	
## Testing: replace "compiled_file_name" with the appropriate name: { vector_add , sum_red , conv , naive_mat_mul , coal_mat_mul , cache_tiled_mat_mul }
	$ ./compiled_file_name

## Results: when running each of the files, it will give the time spent on the GPU-related operations and whether the output of the kernel is correct (compared with a verification function defined on the same file)

## Example:
~$ git clone https://github.com/gara2000/cuda_math_operations.git
~/platform_testing$ cd platform_testing
~$ make clean
rm -f conv sum_red vector_add naive_mat_mul cache_tiled_mat_mul coal_mat_mul
~/platform_testing$ make
nvcc -o conv conv.cu
nvcc -o sum_red sum_red.cu
nvcc -o vector_add vector_add.cu
nvcc -o naive_mat_mul naive_mat_mul.cu
nvcc -o cache_tiled_mat_mul cache_tiled_mat_mul.cu
nvcc -o coal_mat_mul coal_mat_mul.cu
~/platform_testing$ ./vector_add
Elapsed time: 1.80681 ms
Correct results!
~/platform_testing$
