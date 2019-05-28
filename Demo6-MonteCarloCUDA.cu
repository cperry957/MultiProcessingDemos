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
#ifndef XCMIN
#define XCMIN -1.0
#endif
#ifndef XCMAX
#define XCMAX 1.0
#endif
#ifndef YCMIN
#define YCMIN 0.0
#endif
#ifndef YCMAX
#define YCMAX 2.0
#endif
#ifndef RMIN
#define RMIN 0.5
#endif
#ifndef RMAX
#define RMAX 2.0
#endif

// function prototypes:
float		Ranf(float, float);
void		TimeOfDaySeed();

// Monte Carlo (CUDA Kernel) on the device:

__global__  void MonteCarlo(float* xcs, float* ycs, float* rs, float* results)
{
	__shared__ float prods[BLOCKSIZE];

	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

	float xc = xcs[gid];
	float yc = ycs[gid];
	float  r = rs[gid];

	// solve for the intersection using the quadratic formula:
	float a = 2.;
	float b = -2. * (xc + yc);
	float c = xc * xc + yc * yc - r * r;
	float d = b * b - 4. * a * c;
	prods[tnum] = 0.;

	//If d is less than 0., then the circle was completely missed. (Case A).
	if (d >= 0.)
	{
		// hits the circle:
		// get the first intersection:
		d = sqrt(d);
		float t1 = (-b + d) / (2. * a);	// time to intersect the circle
		float t2 = (-b - d) / (2. * a);	// time to intersect the circle
		float tmin = t1 < t2 ? t1 : t2;		// only care about the first intersection

		//If tmin is less than 0., then the circle completely engulfs the laser pointer. (Case B).
		if (tmin >= 0.)
		{
			// where does it intersect the circle?
			float xcir = tmin;
			float ycir = tmin;

			// get the unitized normal vector at the point of intersection:
			float nx = xcir - xc;
			float ny = ycir - yc;
			float nsqrt = sqrt(nx * nx + ny * ny);
			nx /= nsqrt;	// unit vector
			ny /= nsqrt;	// unit vector

			// get the unitized incoming vector:
			float inx = xcir - 0.;
			float iny = ycir - 0.;
			float in = sqrt(inx * inx + iny * iny);
			inx /= in;	// unit vector
			iny /= in;	// unit vector

			// get the outgoing (bounced) vector:
			float dot = inx * nx + iny * ny;
			//float outx = inx - 2. * nx * dot;	// angle of reflection = angle of incidence`
			float outy = iny - 2. * ny * dot;	// angle of reflection = angle of incidence`

			// find out if it hits the infinite plate:
			float t = (0. - ycir) / outy;

			//If t is less than 0., then the reflected beam went up instead of down.  Continue on to the next trial in the for - loop.
			if (t >= 0.)
			{
				//Otherwise, this beam hit the infinite plate. (Case D) Add to the number of hits.
				prods[tnum] = 1.;
			}
		}
	}

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
		results[wgNum] = prods[0];
}

// main program:

int main(int argc, char* argv[])
{
	int dev = findCudaDevice(argc, (const char**)argv);
	TimeOfDaySeed();		// seed the random number generator
	fprintf(stderr, "XCMIN = %lf, XCMAX = %lf - YCMIN = %lf, YCMAX = %lf - RMIN = %lf, RMAX = %lf\n", XCMIN, XCMAX, YCMIN, YCMAX, RMIN, RMAX);

	// allocate host memory:

	float* xcs = new float[SIZE];
	float* ycs = new float[SIZE];
	float* rs = new float[SIZE];

	// fill the random-value arrays:
	for (int i = 0; i < SIZE; i++)
	{
		xcs[i] = Ranf(XCMIN, XCMAX);
		ycs[i] = Ranf(YCMIN, YCMAX);
		rs[i] = Ranf(RMIN, RMAX);
	}

	float* hits = new float[SIZE / BLOCKSIZE];

	// Initialize hits to 0:
	for (int i = 0; i < (SIZE / BLOCKSIZE); i++)
	{
		hits[i] = 0;
	}

	// allocate device memory:

	float* dX, * dY, * dR, * dResults;

	dim3 dimsX(SIZE, 1, 1);
	dim3 dimsY(SIZE, 1, 1);
	dim3 dimsR(SIZE, 1, 1);
	dim3 dimsResults(SIZE / BLOCKSIZE, 1, 1);

	cudaError_t status;
	status = cudaMalloc(reinterpret_cast<void**>(&dX), SIZE * sizeof(float));
	checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void**>(&dY), SIZE * sizeof(float));
	checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void**>(&dR), SIZE * sizeof(float));
	checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void**>(&dResults), (SIZE / BLOCKSIZE) * sizeof(float));
	checkCudaErrors(status);


	// copy host memory to the device:

	status = cudaMemcpy(dX, xcs, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(status);
	status = cudaMemcpy(dY, ycs, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(status);
	status = cudaMemcpy(dR, rs, SIZE * sizeof(float), cudaMemcpyHostToDevice);
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
		MonteCarlo << < grid, threads >> > (dX, dY, dR, dResults);
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

	status = cudaMemcpy(hits, dResults, (SIZE / BLOCKSIZE) * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErrors(status);

	// check the sum :
	double sum = 0.;
	for (int i = 0; i < SIZE / BLOCKSIZE; i++)
	{
		//fprintf(stderr, "hC[%6d] = %10.2f\n", i, hC[i]);
		sum += (double)hits[i];
	}

	double currentProb = sum / (float)(SIZE);
	fprintf(stderr, "\nsum = %lf\n", currentProb);
	printf("%d,%d,%lf,%lf\n", BLOCKSIZE, SIZE, currentProb, megaMultsPerSecond);

	// clean up memory:
	delete[] xcs;
	delete[] ycs;
	delete[] rs;
	delete[] hits;

	status = cudaFree(dX);
	checkCudaErrors(status);
	status = cudaFree(dY);
	checkCudaErrors(status);
	status = cudaFree(dR);
	checkCudaErrors(status);
	status = cudaFree(dResults);
	checkCudaErrors(status);


	return 0;
}

float Ranf(float low, float high)
{
	float r = (float)rand();               // 0 - RAND_MAX
	float t = r / (float)(RAND_MAX);       // 0. - 1.

	return   low + t * (high - low);
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