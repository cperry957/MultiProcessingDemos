// 1. Program header

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <omp.h>

#include "cl.h"
#include "cl_platform.h"

#define ID_AMD		0x1002
#define ID_INTEL	0x8086
#define ID_NVIDIA	0x10de

#ifndef SELECTED_PLATFORM
#define	SELECTED_PLATFORM		1
#endif

#ifndef SELECTED_DEVICE
#define	SELECTED_DEVICE			0
#endif

#ifndef NUM_ELEMENTS
#define NUM_ELEMENTS		8388608
#endif

#ifndef LOCAL_SIZE
#define	LOCAL_SIZE		128
#endif

#define	NUM_WORK_GROUPS		NUM_ELEMENTS/LOCAL_SIZE

const char *		CL_FILE_NAME = { "Demo5-OpenCLArrays.cl" };
const float			TOL = 0.0001f;
const int			C_NUM_ELEMENTS = NUM_ELEMENTS;
const int			MAX_PLATFORMS = 10;
const int			MAX_DEVICES = 10;
void				Wait( cl_command_queue );
void				ArrayMulti();
void				ArrayMultiAdd();
void				ArrayMultiReduce();
void				ArrayMultiReduceDouble();
int				LookAtTheBits( float );
void PrintPlatformDeviceInfo(cl_platform_id*, cl_uint);

int main( int argc, char *argv[ ] )
{
	cl_int status;		
	// returned status from opencl calls
	// test against CL_SUCCESS

	// get the platform id:
	cl_platform_id platform[MAX_PLATFORMS];
	cl_uint platforms;
	status = clGetPlatformIDs(MAX_PLATFORMS, platform, &platforms);
	if( status != CL_SUCCESS )
		fprintf( stderr, "clGetPlatformIDs failed (2) - Status Code:%d\n", (int)status );
	

	//Display Platform Detail
	if (SELECTED_PLATFORM == -1 || SELECTED_DEVICE == -1)
	{
		PrintPlatformDeviceInfo(platform, platforms);
		printf("\nPlease define platform and device indexes from the list above.");
		return 0;
	}

	ArrayMulti();
	ArrayMultiAdd();
	ArrayMultiReduce();
	ArrayMultiReduceDouble();

	return 0;
}

void ArrayMulti()
{
	// see if we can even open the opencl kernel program
	// (no point going on if we can't):

	FILE* fp;
#ifdef WIN32
	errno_t err = fopen_s(&fp, CL_FILE_NAME, "r");
	if (err != 0)
#else
	fp = fopen(CL_FILE_NAME, "r");
	if (fp == NULL)
#endif
	{
		fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
	}

	cl_int status;		
	// returned status from opencl calls
	// test against CL_SUCCESS
	// get the platform id:
	cl_platform_id platform[MAX_PLATFORMS];
	cl_uint platforms;
	status = clGetPlatformIDs(MAX_PLATFORMS, platform, &platforms);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (2) - Status Code:%d\n", (int)status);

	// get the device id:
	cl_device_id device[MAX_DEVICES];
	cl_uint devices;
	status = clGetDeviceIDs(platform[SELECTED_PLATFORM], CL_DEVICE_TYPE_GPU, MAX_DEVICES, device, &devices);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetDeviceIDs failed (2)\n");

	// 2. allocate the host memory buffers:

	float* hA = new float[NUM_ELEMENTS];
	float* hB = new float[NUM_ELEMENTS];
	float* hC = new float[NUM_ELEMENTS];

	// fill the host memory buffers:

	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		hA[i] = hB[i] = (float)sqrt((double)i);
	}

	size_t dataSize = NUM_ELEMENTS * sizeof(float);

	// 3. create an opencl context:

	cl_context context = clCreateContext(NULL, 1, &device[SELECTED_DEVICE], NULL, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateContext failed\n");

	// 4. create an opencl command queue:

	cl_command_queue cmdQueue = clCreateCommandQueue(context, device[SELECTED_DEVICE], 0, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateCommandQueue failed\n");

	// 5. allocate the device memory buffers:

	cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (1)\n");

	cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (2)\n");

	cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (3)\n");

	// 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:

	status = clEnqueueWriteBuffer(cmdQueue, dA, CL_FALSE, 0, dataSize, hA, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");

	status = clEnqueueWriteBuffer(cmdQueue, dB, CL_FALSE, 0, dataSize, hB, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");

	Wait(cmdQueue);

	// 7. read the kernel code from a file:

	fseek(fp, 0, SEEK_END);
	size_t fileSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char* clProgramText = new char[fileSize + 1];		// leave room for '\0'
	size_t n = fread(clProgramText, 1, fileSize, fp);
	clProgramText[fileSize] = '\0';
	fclose(fp);
	if (n != fileSize)
		fprintf(stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", (int)fileSize, CL_FILE_NAME, (int)n);

	// create the text for the kernel program:

	char* strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)strings, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateProgramWithSource failed\n");
	delete[] clProgramText;

	// 8. compile and link the kernel code:

	char optionsString[] = "";
	char* options = optionsString;
	status = clBuildProgram(program, 1, &device[SELECTED_DEVICE], options, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		size_t size;
		clGetProgramBuildInfo(program, device[SELECTED_DEVICE], CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
		cl_char* log = new cl_char[size];
		clGetProgramBuildInfo(program, device[SELECTED_DEVICE], CL_PROGRAM_BUILD_LOG, size, log, NULL);
		fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
		delete[] log;
	}

	// 9. create the kernel object:
	cl_kernel kernel = clCreateKernel(program, "ArrayMult", &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateKernel failed\n");

	// 10. setup the arguments to the kernel object:

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (1)\n");

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (2)\n");

	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (3)\n");


	// 11. enqueue the kernel object for execution:

	size_t globalWorkSize[3] = { NUM_ELEMENTS, 1, 1 };
	size_t localWorkSize[3] = { LOCAL_SIZE,   1, 1 };

	Wait(cmdQueue);
	double time0 = omp_get_wtime();

	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);

	Wait(cmdQueue);
	double time1 = omp_get_wtime();

	// 12. read the results buffer back from the device to the host:

	status = clEnqueueReadBuffer(cmdQueue, dC, CL_TRUE, 0, dataSize, hC, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueReadBuffer failed\n");

	// did it work?

	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		float expected = hA[i] * hB[i];
		if (fabs(hC[i] - expected) > TOL)
		{
			//fprintf(stderr, "%4d: %13.6f * %13.6f wrongly produced %13.6f instead of %13.6f (%13.8f)\n",
			//	i, hA[i], hB[i], hC[i], expected, fabs(hC[i] - expected));
			//fprintf(stderr, "%4d:    0x%08x *    0x%08x wrongly produced    0x%08x instead of    0x%08x\n",
			//	i, LookAtTheBits(hA[i]), LookAtTheBits(hB[i]), LookAtTheBits(hC[i]), LookAtTheBits(expected));
		}
	}

	printf("ArrayMulti,%d,%d,%d,%lf\n", NUM_ELEMENTS, LOCAL_SIZE, NUM_WORK_GROUPS, (double)NUM_ELEMENTS / (time1 - time0) / 1000000000.);

#ifdef WIN32
	Sleep(2000);
#endif


	// 13. clean everything up:

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(dA);
	clReleaseMemObject(dB);
	clReleaseMemObject(dC);

	delete[] hA;
	delete[] hB;
	delete[] hC;
}

void ArrayMultiAdd()
{
	// see if we can even open the opencl kernel program
	// (no point going on if we can't):

	FILE* fp;
#ifdef WIN32
	errno_t err = fopen_s(&fp, CL_FILE_NAME, "r");
	if (err != 0)
#else
	fp = fopen(CL_FILE_NAME, "r");
	if (fp == NULL)
#endif
	{
		fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
	}

	cl_int status;
	// returned status from opencl calls
	// test against CL_SUCCESS
	// get the platform id:
	cl_platform_id platform[MAX_PLATFORMS];
	cl_uint platforms;
	status = clGetPlatformIDs(MAX_PLATFORMS, platform, &platforms);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (2) - Status Code:%d\n", (int)status);

	// get the device id:
	cl_device_id device[MAX_DEVICES];
	cl_uint devices;
	status = clGetDeviceIDs(platform[SELECTED_PLATFORM], CL_DEVICE_TYPE_GPU, MAX_DEVICES, device, &devices);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetDeviceIDs failed (2)\n");

	// 2. allocate the host memory buffers:

	float* hA = new float[NUM_ELEMENTS];
	float* hB = new float[NUM_ELEMENTS];
	float* hC = new float[NUM_ELEMENTS];
	float* hD = new float[NUM_ELEMENTS];

	// fill the host memory buffers:

	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		hA[i] = hB[i] = hC[i] = (float)sqrt((double)i);
	}

	size_t dataSize = NUM_ELEMENTS * sizeof(float);

	// 3. create an opencl context:

	cl_context context = clCreateContext(NULL, 1, &device[SELECTED_DEVICE], NULL, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateContext failed\n");

	// 4. create an opencl command queue:

	cl_command_queue cmdQueue = clCreateCommandQueue(context, device[SELECTED_DEVICE], 0, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateCommandQueue failed\n");

	// 5. allocate the device memory buffers:

	cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (1)\n");

	cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (2)\n");

	cl_mem dC = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (3)\n");

	cl_mem dD = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (4)\n");

	// 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:

	status = clEnqueueWriteBuffer(cmdQueue, dA, CL_FALSE, 0, dataSize, hA, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");

	status = clEnqueueWriteBuffer(cmdQueue, dB, CL_FALSE, 0, dataSize, hB, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");

	status = clEnqueueWriteBuffer(cmdQueue, dC, CL_FALSE, 0, dataSize, hC, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (3)\n");

	Wait(cmdQueue);

	// 7. read the kernel code from a file:

	fseek(fp, 0, SEEK_END);
	size_t fileSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char* clProgramText = new char[fileSize + 1];		// leave room for '\0'
	size_t n = fread(clProgramText, 1, fileSize, fp);
	clProgramText[fileSize] = '\0';
	fclose(fp);
	if (n != fileSize)
		fprintf(stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", (int)fileSize, CL_FILE_NAME, (int)n);

	// create the text for the kernel program:

	char* strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)strings, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateProgramWithSource failed\n");
	delete[] clProgramText;

	// 8. compile and link the kernel code:

	char optionsString[] = "";
	char* options = optionsString;
	status = clBuildProgram(program, 1, &device[SELECTED_DEVICE], options, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		size_t size;
		clGetProgramBuildInfo(program, device[SELECTED_DEVICE], CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
		cl_char* log = new cl_char[size];
		clGetProgramBuildInfo(program, device[SELECTED_DEVICE], CL_PROGRAM_BUILD_LOG, size, log, NULL);
		fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
		delete[] log;
	}

	// 9. create the kernel object:
	cl_kernel kernel = clCreateKernel(program, "ArrayMultAdd", &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateKernel failed\n");

	// 10. setup the arguments to the kernel object:

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (1)\n");

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (2)\n");

	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (3)\n");

	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dD);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (4)\n");


	// 11. enqueue the kernel object for execution:

	size_t globalWorkSize[3] = { NUM_ELEMENTS, 1, 1 };
	size_t localWorkSize[3] = { LOCAL_SIZE,   1, 1 };

	Wait(cmdQueue);
	double time0 = omp_get_wtime();

	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);

	Wait(cmdQueue);
	double time1 = omp_get_wtime();

	// 12. read the results buffer back from the device to the host:

	status = clEnqueueReadBuffer(cmdQueue, dD, CL_TRUE, 0, dataSize, hD, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueReadBuffer failed\n");

	// did it work?

	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		float expected = (hA[i] * hB[i]) + hC[i];
		if (fabs(hD[i] - expected) > TOL)
		{
			//fprintf(stderr, "%4d: (%13.6f * %13.6f) + %13.6f wrongly produced %13.6f instead of %13.6f (%13.8f)\n",
			//	i, hA[i], hB[i], hC[i], hD[i], expected, fabs(hD[i] - expected));
			//fprintf(stderr, "%4d:    0x%08x *    0x%08x wrongly produced    0x%08x instead of    0x%08x\n",
			//	i, LookAtTheBits(hA[i]), LookAtTheBits(hB[i]), LookAtTheBits(hC[i]), LookAtTheBits(expected));
		}
	}

	printf("ArrayMultiAdd,%d,%d,%d,%lf\n", NUM_ELEMENTS, LOCAL_SIZE, NUM_WORK_GROUPS, (double)NUM_ELEMENTS / (time1 - time0) / 1000000000.);

#ifdef WIN32
	Sleep(2000);
#endif


	// 13. clean everything up:

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(dA);
	clReleaseMemObject(dB);
	clReleaseMemObject(dC);
	clReleaseMemObject(dD);

	delete[] hA;
	delete[] hB;
	delete[] hC;
	delete[] hD;
}

void ArrayMultiReduce()
{
	// see if we can even open the opencl kernel program
	// (no point going on if we can't):

	FILE* fp;
#ifdef WIN32
	errno_t err = fopen_s(&fp, CL_FILE_NAME, "r");
	if (err != 0)
#else
	fp = fopen(CL_FILE_NAME, "r");
	if (fp == NULL)
#endif
	{
		fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
	}

	cl_int status;
	// returned status from opencl calls
	// test against CL_SUCCESS
	// get the platform id:
	cl_platform_id platform[MAX_PLATFORMS];
	cl_uint platforms;
	status = clGetPlatformIDs(MAX_PLATFORMS, platform, &platforms);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (2) - Status Code:%d\n", (int)status);

	// get the device id:
	cl_device_id device[MAX_DEVICES];
	cl_uint devices;
	status = clGetDeviceIDs(platform[SELECTED_PLATFORM], CL_DEVICE_TYPE_GPU, MAX_DEVICES, device, &devices);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetDeviceIDs failed (2)\n");

	// 2. allocate the host memory buffers:
	size_t numWorkGroups = NUM_ELEMENTS / LOCAL_SIZE;
	float* hA = new float[NUM_ELEMENTS];
	float* hB = new float[NUM_ELEMENTS];
	float* hC = new float[numWorkGroups];
	size_t abSize = NUM_ELEMENTS * sizeof(float);
	size_t cSize = numWorkGroups * sizeof(float);

	// fill the host memory buffers:
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		hA[i] = hB[i] = (float)sqrt((double)i);
	}

	// 3. create an opencl context:

	cl_context context = clCreateContext(NULL, 1, &device[SELECTED_DEVICE], NULL, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateContext failed\n");

	// 4. create an opencl command queue:

	cl_command_queue cmdQueue = clCreateCommandQueue(context, device[SELECTED_DEVICE], 0, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateCommandQueue failed\n");

	// 5. allocate the device memory buffers:

	cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, abSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (1)\n");

	cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY, abSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (2)\n");

	cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, cSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (3)\n");

	// 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:

	status = clEnqueueWriteBuffer(cmdQueue, dA, CL_FALSE, 0, abSize, hA, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");

	status = clEnqueueWriteBuffer(cmdQueue, dB, CL_FALSE, 0, abSize, hB, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");

	Wait(cmdQueue);

	// 7. read the kernel code from a file:

	fseek(fp, 0, SEEK_END);
	size_t fileSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char* clProgramText = new char[fileSize + 1];		// leave room for '\0'
	size_t n = fread(clProgramText, 1, fileSize, fp);
	clProgramText[fileSize] = '\0';
	fclose(fp);
	if (n != fileSize)
		fprintf(stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", (int)fileSize, CL_FILE_NAME, (int)n);

	// create the text for the kernel program:

	char* strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)strings, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateProgramWithSource failed\n");
	delete[] clProgramText;

	// 8. compile and link the kernel code:

	char optionsString[] = "";
	char* options = optionsString;
	status = clBuildProgram(program, 1, &device[SELECTED_DEVICE], options, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		size_t size;
		clGetProgramBuildInfo(program, device[SELECTED_DEVICE], CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
		cl_char* log = new cl_char[size];
		clGetProgramBuildInfo(program, device[SELECTED_DEVICE], CL_PROGRAM_BUILD_LOG, size, log, NULL);
		fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
		delete[] log;
	}

	// 9. create the kernel object:
	cl_kernel kernel = clCreateKernel(program, "ArrayMultReduce", &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateKernel failed\n");

	// 10. setup the arguments to the kernel object:

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (1)\n");

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (2)\n");

	status = clSetKernelArg(kernel, 2, LOCAL_SIZE * sizeof(float), NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (3)\n");

	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dC);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (4)\n");


	// 11. enqueue the kernel object for execution:

	size_t globalWorkSize[3] = { NUM_ELEMENTS, 1, 1 };
	size_t localWorkSize[3] = { LOCAL_SIZE,   1, 1 };

	Wait(cmdQueue);
	double time0 = omp_get_wtime();

	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);

	Wait(cmdQueue);
	double time1 = omp_get_wtime();

	// 12. read the results buffer back from the device to the host:

	status = clEnqueueReadBuffer(cmdQueue, dC, CL_TRUE, 0, numWorkGroups * sizeof(float), hC, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueReadBuffer failed\n");
	Wait(cmdQueue);

	float sum = 0.;
	for (int i = 0; i < numWorkGroups; i++)
	{
		sum += hC[i];
	}

	// did it work?
	float expected = 0.;
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		expected += (hA[i] * hB[i]);
	}

	if (fabs(sum - expected) > TOL)
	{
		fprintf(stderr, "%13.6f wrongly produced instead of %13.6f (%13.8f)\n",
			sum, expected, fabs(sum - expected));
	}

	printf("ArrayMultiReduce,%d,%d,%d,%lf\n", NUM_ELEMENTS, LOCAL_SIZE, NUM_WORK_GROUPS, (double)NUM_ELEMENTS / (time1 - time0) / 1000000000.);


#ifdef WIN32
	Sleep(2000);
#endif


	// 13. clean everything up:

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(dA);
	clReleaseMemObject(dB);
	clReleaseMemObject(dC);

	delete[] hA;
	delete[] hB;
	delete[] hC;
}

void ArrayMultiReduceDouble()
{
	// see if we can even open the opencl kernel program
	// (no point going on if we can't):

	FILE* fp;
#ifdef WIN32
	errno_t err = fopen_s(&fp, CL_FILE_NAME, "r");
	if (err != 0)
#else
	fp = fopen(CL_FILE_NAME, "r");
	if (fp == NULL)
#endif
	{
		fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
	}

	cl_int status;
	// returned status from opencl calls
	// test against CL_SUCCESS
	// get the platform id:
	cl_platform_id platform[MAX_PLATFORMS];
	cl_uint platforms;
	status = clGetPlatformIDs(MAX_PLATFORMS, platform, &platforms);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (2) - Status Code:%d\n", (int)status);

	// get the device id:
	cl_device_id device[MAX_DEVICES];
	cl_uint devices;
	status = clGetDeviceIDs(platform[SELECTED_PLATFORM], CL_DEVICE_TYPE_GPU, MAX_DEVICES, device, &devices);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetDeviceIDs failed (2)\n");

	// 2. allocate the host memory buffers:
	size_t numWorkGroups = NUM_ELEMENTS / LOCAL_SIZE;
	double* hA = new double[NUM_ELEMENTS];
	double* hB = new double[NUM_ELEMENTS];
	double* hC = new double[numWorkGroups];
	size_t abSize = NUM_ELEMENTS * sizeof(double);
	size_t cSize = numWorkGroups * sizeof(double);

	// fill the host memory buffers:
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		hA[i] = hB[i] = sqrt((double)i);
	}

	// 3. create an opencl context:

	cl_context context = clCreateContext(NULL, 1, &device[SELECTED_DEVICE], NULL, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateContext failed\n");

	// 4. create an opencl command queue:

	cl_command_queue cmdQueue = clCreateCommandQueue(context, device[SELECTED_DEVICE], 0, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateCommandQueue failed\n");

	// 5. allocate the device memory buffers:

	cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, abSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (1)\n");

	cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY, abSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (2)\n");

	cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, cSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (3)\n");

	// 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:

	status = clEnqueueWriteBuffer(cmdQueue, dA, CL_FALSE, 0, abSize, hA, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");

	status = clEnqueueWriteBuffer(cmdQueue, dB, CL_FALSE, 0, abSize, hB, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");

	Wait(cmdQueue);

	// 7. read the kernel code from a file:

	fseek(fp, 0, SEEK_END);
	size_t fileSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char* clProgramText = new char[fileSize + 1];		// leave room for '\0'
	size_t n = fread(clProgramText, 1, fileSize, fp);
	clProgramText[fileSize] = '\0';
	fclose(fp);
	if (n != fileSize)
		fprintf(stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", (int)fileSize, CL_FILE_NAME, (int)n);

	// create the text for the kernel program:

	char* strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)strings, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateProgramWithSource failed\n");
	delete[] clProgramText;

	// 8. compile and link the kernel code:

	char optionsString[] = "";
	char* options = optionsString;
	status = clBuildProgram(program, 1, &device[SELECTED_DEVICE], options, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		size_t size;
		clGetProgramBuildInfo(program, device[SELECTED_DEVICE], CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
		cl_char* log = new cl_char[size];
		clGetProgramBuildInfo(program, device[SELECTED_DEVICE], CL_PROGRAM_BUILD_LOG, size, log, NULL);
		fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
		delete[] log;
	}

	// 9. create the kernel object:
	cl_kernel kernel = clCreateKernel(program, "ArrayMultReduceDouble", &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateKernel failed\n");

	// 10. setup the arguments to the kernel object:

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (1)\n");

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (2)\n");

	status = clSetKernelArg(kernel, 2, LOCAL_SIZE * sizeof(double), NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (3)\n");

	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dC);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (4)\n");


	// 11. enqueue the kernel object for execution:

	size_t globalWorkSize[3] = { NUM_ELEMENTS, 1, 1 };
	size_t localWorkSize[3] = { LOCAL_SIZE,   1, 1 };

	Wait(cmdQueue);
	double time0 = omp_get_wtime();

	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);

	Wait(cmdQueue);
	double time1 = omp_get_wtime();

	// 12. read the results buffer back from the device to the host:

	status = clEnqueueReadBuffer(cmdQueue, dC, CL_TRUE, 0, numWorkGroups * sizeof(double), hC, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueReadBuffer failed\n");
	Wait(cmdQueue);

	double sum = 0.;
	for (int i = 0; i < numWorkGroups; i++)
	{
		sum += hC[i];
	}

	// did it work?
	double expected = 0.;
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		expected += (hA[i] * hB[i]);
	}

	if (fabs(sum - expected) > TOL)
	{
		fprintf(stderr, "%13.6f wrongly produced instead of %13.6f (%13.8f)\n",
			sum, expected, fabs(sum - expected));
	}

	printf("ArrayMultiReduceDouble,%d,%d,%d,%lf\n", NUM_ELEMENTS, LOCAL_SIZE, NUM_WORK_GROUPS, (double)NUM_ELEMENTS / (time1 - time0) / 1000000000.);


#ifdef WIN32
	Sleep(2000);
#endif


	// 13. clean everything up:

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(dA);
	clReleaseMemObject(dB);
	clReleaseMemObject(dC);

	delete[] hA;
	delete[] hB;
	delete[] hC;
}

int LookAtTheBits( float fp )
{
	int *ip = (int *)&fp;
	return *ip;
}


// wait until all queued tasks have taken place:

void Wait( cl_command_queue queue )
{
      cl_event wait;
      cl_int      status;

      status = clEnqueueMarker( queue, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clEnqueueMarker failed\n" );

      status = clWaitForEvents( 1, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clWaitForEvents failed\n" );
}

void PrintPlatformDeviceInfo(cl_platform_id* platforms, cl_uint numPlatforms) {
	cl_int status;
	for (int i = 0; i < (int)numPlatforms; i++)
	{
		fprintf(stderr, "Platform #%d:\n", i);
		size_t size;
		char* str;

		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &size);
		str = new char[size];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, size, str, NULL);
		fprintf(stderr, "\tName    = '%s'\n", str);
		delete[] str;

		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
		str = new char[size];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, size, str, NULL);
		fprintf(stderr, "\tVendor  = '%s'\n", str);
		delete[] str;

		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 0, NULL, &size);
		str = new char[size];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, size, str, NULL);
		fprintf(stderr, "\tVersion = '%s'\n", str);
		delete[] str;

		clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 0, NULL, &size);
		str = new char[size];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, size, str, NULL);
		fprintf(stderr, "\tProfile = '%s'\n", str);
		delete[] str;


		// find out how many devices are attached to each platform and get their ids:
		cl_uint numDevices;
		cl_device_id* devices;
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetDeviceIDs failed (2)\n");

		fprintf(stderr, "\tNumber of Devices = %d\n", numDevices);

		devices = new cl_device_id[numDevices];
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetDeviceIDs failed (2)\n");

		for (int j = 0; j < (int)numDevices; j++)
		{
			fprintf(stderr, "\tDevice #%d:\n", j);
			size_t size;
			cl_device_type type;
			cl_uint ui;
			size_t sizes[3] = { 0, 0, 0 };

			clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
			fprintf(stderr, "\t\tType = 0x%04x = ", (unsigned int)type);
			switch (type)
			{
			case CL_DEVICE_TYPE_CPU:
				fprintf(stderr, "CL_DEVICE_TYPE_CPU\n");
				break;
			case CL_DEVICE_TYPE_GPU:
				fprintf(stderr, "CL_DEVICE_TYPE_GPU\n");
				break;
			case CL_DEVICE_TYPE_ACCELERATOR:
				fprintf(stderr, "CL_DEVICE_TYPE_ACCELERATOR\n");
				break;
			default:
				fprintf(stderr, "Other...\n");
				break;
			}

			clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR_ID, sizeof(ui), &ui, NULL);
			fprintf(stderr, "\t\tDevice Vendor ID = 0x%04x ", ui);
			switch (ui)
			{
			case ID_AMD:
				fprintf(stderr, "(AMD)\n");
				break;
			case ID_INTEL:
				fprintf(stderr, "(Intel)\n");
				break;
			case ID_NVIDIA:
				fprintf(stderr, "(NVIDIA)\n");
				break;
			default:
				fprintf(stderr, "(?)\n");
			}

			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ui), &ui, NULL);
			fprintf(stderr, "\t\tDevice Maximum Compute Units = %d\n", ui);

			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(ui), &ui, NULL);
			fprintf(stderr, "\t\tDevice Maximum Work Item Dimensions = %d\n", ui);

			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes, NULL);
			fprintf(stderr, "\t\tDevice Maximum Work Item Sizes = %d x %d x %d\n", (int)sizes[0], (int)sizes[1], (int)sizes[2]);

			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size), &size, NULL);
			fprintf(stderr, "\t\tDevice Maximum Work Group Size = %d\n", (int)size);

			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ui), &ui, NULL);
			fprintf(stderr, "\t\tDevice Maximum Clock Frequency = %d MHz\n", ui);
		}
	}
}