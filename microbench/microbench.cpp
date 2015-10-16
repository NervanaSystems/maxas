// microbench.cpp : Defines the entry point for the console application.
//

// nvcc -l cuda -o microbench microbench.cpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cudaProfiler.h>

CUcontext hContext = 0;

#define CUDA_CHECK( fn ) do { \
		CUresult status = (fn); \
		if ( CUDA_SUCCESS != status ) { \
			const char* errstr; \
			cuGetErrorString(status, &errstr); \
			printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
			if (hContext) cuCtxDestroy(hContext); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)


int main(int argc, char* argv[])
{
	//int iTest = 2896;
	//while (iTest < 0x7fff)
	//{
	//	int iResult = iTest * iTest;
	//	float fTest = (float)iTest;
	//	int fResult = (int)(fTest * fTest);

	//	printf("i*i:%08x f*f:%08x\n", iResult, fResult);

	//	iTest += 0x0800;
	//}
	//exit(0);

	char deviceName[32];
	int devCount, ordinal, major, minor;
	CUdevice  hDevice;

	// Initialize the Driver API and find a device
	CUDA_CHECK( cuInit(0) );
	CUDA_CHECK( cuDeviceGetCount(&devCount) );
	for (ordinal = 0; ordinal < devCount; ordinal++)
	{
		CUDA_CHECK( cuDeviceGet(&hDevice, ordinal) );
		CUDA_CHECK( cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice) );
		CUDA_CHECK( cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice) );
		CUDA_CHECK( cuDeviceGetName(deviceName, sizeof(deviceName), hDevice) );
		if (major >= 5 && minor >= 2)
		{
			printf("Using: Id:%d %s (%d.%d)\n\n", ordinal, deviceName, major, minor);
			break;
		}
	}
	if (ordinal == devCount)
	{
		printf("No compute 5.0 device found, exiting.\n");
		exit(EXIT_FAILURE);
	}

	// First command line arg is the type: internal (CS2R) or external (cuEventElapsedTime) timing
	int internalTiming = 1;
	if (argc > 1)
		internalTiming = strcmp(argv[1], "i") == 0 ? 1 : 0;

	// Second command line arg is the number of blocks
	int blocks = 1;
	if (argc > 2)
		blocks = atoi(argv[2]);
	if (blocks < 1)
		blocks = 1;

	// Third command line arg is the number of threads
	int threads = 128;
	if (argc > 3)
		threads = atoi(argv[3]);
	if (threads > 1024 || threads < 32)
		threads = 128;
	threads &= -32;

	// Forth command line arg:
	double fops = 1.0;
	int lanes = 1;
	if (argc > 4)
	{
		if (internalTiming)
		{
			// The number of lanes to print for each warp
			lanes = atoi(argv[4]);
			if (lanes > 32 || lanes < 1)
				lanes = 1;
		}
		else
			// The number of floating point operations in a full kernel launch
			fops = atof(argv[4]);
	}

	// Fifth command line arg is the repeat count for benchmarking
	int repeat = 1;
	if (argc > 5)
		repeat = atoi(argv[5]);
	if (repeat > 1000 || repeat < 1)
		repeat = 1;

	// threads = total number of threads
	size_t size = sizeof(int) * threads * blocks;

	// Setup our input and output buffers
	int* dataIn  = (int*)malloc(size);
	int* dataOut = (int*)malloc(size);
	int* clocks  = (int*)malloc(size);
	memset(dataIn, 0, size);

	CUmodule hModule;
	CUfunction hKernel;
	CUevent hStart, hStop;
	CUdeviceptr devIn, devOut, devClocks;

	// Init our context and device memory buffers
	CUDA_CHECK( cuCtxCreate(&hContext, 0, hDevice) );
	CUDA_CHECK( cuMemAlloc(&devIn, size) );
	CUDA_CHECK( cuMemAlloc(&devOut, size) );
	CUDA_CHECK( cuMemAlloc(&devClocks, size) );
	CUDA_CHECK( cuMemcpyHtoD(devIn, dataIn, size) );
	CUDA_CHECK( cuMemsetD8(devOut, 0, size) );
	CUDA_CHECK( cuMemsetD8(devClocks, 0, size) );

	CUDA_CHECK( cuEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC) );
	CUDA_CHECK( cuEventCreate(&hStop,  CU_EVENT_BLOCKING_SYNC) );

	// Load our kernel
	CUDA_CHECK( cuModuleLoad(&hModule, "microbench.cubin") );
	CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, "microbench") );

	// Setup the params
	void* params[] = { &devOut, &devClocks, &devIn };
	float ms = 0;

	// Warm up the clock (unless under nsight)
	if (!getenv("NSIGHT_LAUNCHED")) // NSIGHT_CUDA_ANALYSIS NSIGHT_CUDA_DEBUGGER
		for (int i = 0; i < repeat; i++)
			CUDA_CHECK( cuLaunchKernel(hKernel, blocks, 1, 1, threads, 1, 1, 0, 0, params, 0) );

	// Launch the kernel
	CUDA_CHECK( cuEventRecord(hStart, NULL) );
	//CUDA_CHECK( cuProfilerStart() );
	CUDA_CHECK( cuLaunchKernel(hKernel, blocks, 1, 1, threads, 1, 1, 0, 0, params, 0) );
	//CUDA_CHECK( cuProfilerStop() );
	CUDA_CHECK( cuEventRecord(hStop, NULL) );
	CUDA_CHECK( cuEventSynchronize(hStop) );
	CUDA_CHECK( cuEventElapsedTime(&ms, hStart, hStop) );

	//CUDA_CHECK( cuCtxSynchronize() );

	// Get back our results from each kernel
	CUDA_CHECK( cuMemcpyDtoH(dataOut, devOut, size) );
	CUDA_CHECK( cuMemcpyDtoH(clocks, devClocks, size) );

	// Cleanup and shutdown of cuda
	CUDA_CHECK( cuEventDestroy(hStart) );
	CUDA_CHECK( cuEventDestroy(hStop) );
	CUDA_CHECK( cuModuleUnload(hModule) );
	CUDA_CHECK( cuMemFree(devIn) );
	CUDA_CHECK( cuMemFree(devOut) );
	CUDA_CHECK( cuMemFree(devClocks) );
	CUDA_CHECK( cuCtxDestroy(hContext) );
	hContext = 0;

	// When using just one block, print out the internal timing data
	if (internalTiming)
	{
		int count = 0, total = 0, min = 999999, max = 0;

		int* clocks_p  = clocks;
		int* dataOut_p = dataOut;

		// Loop over and print results
		for (int blk = 0; blk < blocks; blk++)
		{
			float *fDataOut = reinterpret_cast<float*>(dataOut_p);

			for(int tid = 0; tid < threads; tid += 32)
			{
				// Sometimes we want data on each thread, sometimes just one sample per warp is fine
				for (int lane = 0; lane < lanes; lane++)
					printf("b:%02d w:%03d t:%04d l:%02d clocks:%08d out:%08x\n", blk, tid/32, tid, lane, clocks_p[tid+lane], dataOut_p[tid+lane]); // %04u

				count++;
				total += clocks_p[tid];
				if (clocks_p[tid] < min) min = clocks_p[tid];
				if (clocks_p[tid] > max) max = clocks_p[tid];
			}
			clocks_p  += threads;
			dataOut_p += threads;
		}
		printf("average: %.3f, min %d, max: %d\n", (float)total/count, min, max);
	}
	else
	{
		// For more than one block we're testing throughput and want external timing data
		printf("MilliSecs: %.3f, GFLOPS: %.3f\n", ms, fops / (ms * 1000000.0));
	}
	// And free up host memory
	free(dataIn); free(dataOut); free(clocks);

	return 0;
}
