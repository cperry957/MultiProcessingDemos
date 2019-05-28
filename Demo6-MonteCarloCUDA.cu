// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef BLOCKSIZE
#define BLOCKSIZE		32		// number of threads per block
#endif

#ifndef SIZE
#define SIZE			1*1024*1024	// array size
#endif

#ifndef NUMTRIALS
#define NUMTRIALS		100		// to make the timing more accurate
#endif

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif

// ranges for the random numbers:
const float XCMIN = -1.0;
const float XCMAX = 1.0;
const float YCMIN = 0.0;
const float YCMAX = 2.0;
const float RMIN = 0.5;
const float RMAX = 2.0;

// function prototypes:
float		Ranf(float, float);
int		Ranf(int, int);
void		TimeOfDaySeed();

// Monte Carlo (CUDA Kernel) on the device:

__global__  void MonteCarlo(float* A, float* B, float* C, float* D)
{
	__shared__ float prods[BLOCKSIZE];

	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

	//prods[tnum] = A[gid] * B[gid];
	prods[tnum] = 1.;

	for (int offset = 1; offset < numItems; offset *= 2)
	{
		int mask = 2 * offset - 1;
		__syncthreads();
		if ((tnum & mask) == 0)
		{
			prods[tnum] += prods[tnum + offset];
		}
	}

	__syncthreads();
	if (tnum == 0)
		D[wgNum] = prods[0];
}

// main program:

int main(int argc, char* argv[])
{
	int dev = findCudaDevice(argc, (const char**)argv);
	TimeOfDaySeed();		// seed the random number generator

	// allocate host memory:

	float* xcs = new float[SIZE];
	float* ycs = new float[SIZE];
	float* rs = new float[SIZE];

	float* hits = new float[SIZE / BLOCKSIZE];

	// fill the random-value arrays:
	for (int i = 0; i < SIZE; i++)
	{
		xcs[i] = Ranf(XCMIN, XCMAX);
		ycs[i] = Ranf(YCMIN, YCMAX);
		rs[i] = Ranf(RMIN, RMAX);
	}

	// get ready to record the maximum performance and the probability:
	float maxPerformance = 0.;      // must be declared outside the NUMTRIES loop
	float currentProb;              // must be declared outside the NUMTRIES loop

	// allocate device memory:

	float* dA, * dB, * dC, * dD;

	dim3 dimsA(SIZE, 1, 1);
	dim3 dimsB(SIZE, 1, 1);
	dim3 dimsC(SIZE, 1, 1);
	dim3 dimsD(SIZE / BLOCKSIZE, 1, 1);

	//__shared__ float prods[SIZE/BLOCKSIZE];


	cudaError_t status;
	status = cudaMalloc(reinterpret_cast<void**>(&dA), SIZE * sizeof(float));
	checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void**>(&dB), SIZE * sizeof(float));
	checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void**>(&dC), SIZE * sizeof(float));
	checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void**>(&dD), (SIZE / BLOCKSIZE) * sizeof(float));
	checkCudaErrors(status);


	// copy host memory to the device:

	status = cudaMemcpy(dA, xcs, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(status);
	status = cudaMemcpy(dB, ycs, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(status);
	status = cudaMemcpy(dC, rs, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(status);

	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1);
	dim3 grid(SIZE / threads.x, 1, 1);

	// Create and start timer

	cudaDeviceSynchronize();

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate(&start);
	checkCudaErrors(status);
	status = cudaEventCreate(&stop);
	checkCudaErrors(status);

	// record the start event:

	status = cudaEventRecord(start, NULL);
	checkCudaErrors(status);

	// execute the kernel:

	for (int t = 0; t < NUMTRIALS; t++)
	{
		MonteCarlo << < grid, threads >> > (dA, dB, dC, dD);
	}

	// record the stop event:

	status = cudaEventRecord(stop, NULL);
	checkCudaErrors(status);

	// wait for the stop event to complete:

	status = cudaEventSynchronize(stop);
	checkCudaErrors(status);

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime(&msecTotal, start, stop);
	checkCudaErrors(status);

	// compute and print the performance

	double secondsTotal = 0.001 * (double)msecTotal;
	double multsPerSecond = (float)SIZE * (float)NUMTRIALS / secondsTotal;
	double megaMultsPerSecond = multsPerSecond / 1000000.;
	fprintf(stderr, "Array Size = %10d, MegaMultReductions/Second = %10.2lf\n", SIZE, megaMultsPerSecond);

	// copy result from the device to the host:

	status = cudaMemcpy(hits, dD, (SIZE / BLOCKSIZE) * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErrors(status);

	// check the sum :

	double sum = 0.;
	for (int i = 0; i < SIZE / BLOCKSIZE; i++)
	{
		//fprintf(stderr, "hC[%6d] = %10.2f\n", i, hC[i]);
		sum += (double)hits[i];
	}
	fprintf(stderr, "\nsum = %10.2lf\n", sum);

	// clean up memory:
	delete[] xcs;
	delete[] ycs;
	delete[] rs;
	delete[] hits;

	status = cudaFree(dA);
	checkCudaErrors(status);
	status = cudaFree(dB);
	checkCudaErrors(status);
	status = cudaFree(dC);
	checkCudaErrors(status);
	status = cudaFree(dD);
	checkCudaErrors(status);


	return 0;
}

float Ranf(float low, float high)
{
	float r = (float)rand();               // 0 - RAND_MAX
	float t = r / (float)RAND_MAX;       // 0. - 1.

	return   low + t * (high - low);
}

int
Ranf(int ilow, int ihigh)
{
	float low = (float)ilow;
	float high = ceil((float)ihigh);

	return (int)Ranf(low, high);
}

void TimeOfDaySeed()
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time(&timer);
	double seconds = difftime(timer, mktime(&y2k));
	unsigned int seed = (unsigned int)(1000. * seconds);    // milliseconds
	srand(seed);
}