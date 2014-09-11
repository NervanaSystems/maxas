// sgemm.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>

CUcontext      hContext = 0;
cublasHandle_t hCublas  = 0;

float assemblySgemm(CUdeviceptr devC, CUdeviceptr devA, CUdeviceptr devB, int N, CUevent hStart, CUevent hStop, int repeat = 1, int printVars = 0);
void gflops(const char* ident, int N, float ms, int repeat);
void test(float* C, float* T, int N, size_t size);

#define CUDA_CHECK( fn ) do { \
		CUresult status = (fn); \
		if ( CUDA_SUCCESS != status ) { \
			const char* errstr; \
			cuGetErrorString(status, &errstr); \
			printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
			if (hCublas)  cublasDestroy(hCublas); \
			if (hContext) cuCtxDestroy(hContext); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)

#define CUBLAS_CHECK( fn ) do { \
		cublasStatus_t status = (fn); \
		if ( CUBLAS_STATUS_SUCCESS != status ) { \
			printf("Cublas Failure (line %d of file %s):\n\t%s returned %d\n", __LINE__, __FILE__, #fn, status); \
			if (hCublas)  cublasDestroy(hCublas); \
			if (hContext) cuCtxDestroy(hContext); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)

int main(int argc, char* argv[])
{
	char deviceName[32];
	int count, ordinal, major, minor;
	CUdevice  hDevice;
	CUevent hStart, hStop;
	CUdeviceptr devA, devB, devC, devT;

	// Initialize the Driver API and find a device
	CUDA_CHECK( cuInit(0) );
	CUDA_CHECK( cuDeviceGetCount(&count) );
	for (ordinal = 0; ordinal < count; ordinal++)
	{
		CUDA_CHECK( cuDeviceGet(&hDevice, ordinal) );
		CUDA_CHECK( cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice) );
		CUDA_CHECK( cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice) );
		CUDA_CHECK( cuDeviceGetName(deviceName, sizeof(deviceName), hDevice) );
		if (major >= 5)
		{
			printf("Using: Id:%d %s (%d.%d)\n\n", ordinal, deviceName, major, minor);
			break;
		}
	}
	if (ordinal == count)
	{
		printf("No compute 5.0 device found, exiting.\n");
		exit(EXIT_FAILURE);
	}

	// First command line arg is the size of N divided by 128
	int thread128 = 40;
	if (argc > 1)
		thread128 = atoi(argv[1]);
	if (thread128 > 40 || thread128 < 1)
		thread128 = 1;

	// Second command line arg is the repeat count for benchmarking
	int repeat = 1;
	if (argc > 2)
		repeat = atoi(argv[2]);
	if (repeat > 1000 || repeat < 1)
		repeat = 1;

	// Third command line arg is for printf debugging
	int printVars = 0;
	if (argc > 3)
		printVars = atoi(argv[3]);
	if (printVars > 100 || printVars < 1)
		printVars = 0;

	int N = thread128 * 128;
	size_t size = sizeof(float) * N * N;
	float alpha = 1, beta = 0, ms = 1;

	float* A = (float*)malloc(size);
	float* B = (float*)malloc(size);
	float* C = (float*)malloc(size);
	float* T = (float*)malloc(size);

	srand((unsigned int)time(0));
	for(int i = 0; i < N * N; i++)
	{
		A[i] = (float)rand() / (float)RAND_MAX;
		B[i] = (float)rand() / (float)RAND_MAX;
	}

	CUDA_CHECK( cuCtxCreate(&hContext, 0, hDevice) );
	CUBLAS_CHECK( cublasCreate(&hCublas) );
	
	CUDA_CHECK( cuEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC) );
	CUDA_CHECK( cuEventCreate(&hStop,  CU_EVENT_BLOCKING_SYNC) );

	CUDA_CHECK( cuMemAlloc(&devA, size) );
	CUDA_CHECK( cuMemAlloc(&devB, size) );
	CUDA_CHECK( cuMemAlloc(&devC, size) );
	CUDA_CHECK( cuMemAlloc(&devT, size) );
	
	CUDA_CHECK( cuMemcpyHtoD(devA, A, size) );
	CUDA_CHECK( cuMemcpyHtoD(devB, B, size) );
	CUDA_CHECK( cuMemsetD8(devC, 0, size) );
	CUDA_CHECK( cuMemsetD8(devT, 0, size) );
	
	// Warm up the clock (unless under nsight)
	if (!getenv("NSIGHT_LAUNCHED")) // NSIGHT_CUDA_ANALYSIS NSIGHT_CUDA_DEBUGGER 
		CUBLAS_CHECK( cublasSgemm(hCublas, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, reinterpret_cast<float*>(devA), N, reinterpret_cast<float*>(devB), N, &beta, reinterpret_cast<float*>(devT), N) );

	// Launch our kernel
	ms = assemblySgemm(devC, devA, devB, N, hStart, hStop, repeat, printVars);
	gflops("MaxAs ", N, ms, repeat);

	// Run cublas again for the same repeat count for comparison
	CUDA_CHECK( cuEventRecord(hStart, NULL) );
	for (int i = 0; i < repeat; i++)
		CUBLAS_CHECK( cublasSgemm(hCublas, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, reinterpret_cast<float*>(devA), N, reinterpret_cast<float*>(devB), N, &beta, reinterpret_cast<float*>(devT), N) );
	CUDA_CHECK( cuEventRecord(hStop, NULL) );
	CUDA_CHECK( cuEventSynchronize(hStop) );
	CUDA_CHECK( cuEventElapsedTime(&ms, hStart, hStop) );
	gflops("Cublas", N, ms, repeat);

	// Get back our results from each kernel
	CUDA_CHECK( cuMemcpyDtoH(C, devC, size) );
	CUDA_CHECK( cuMemcpyDtoH(T, devT, size) );
	
	// Cleanup and shutdown of cuda
	CUDA_CHECK( cuMemFree(devA) );
	CUDA_CHECK( cuMemFree(devB) );
	CUDA_CHECK( cuMemFree(devC) );
	CUDA_CHECK( cuMemFree(devT) );

	CUDA_CHECK( cuEventDestroy(hStart) );
	CUDA_CHECK( cuEventDestroy(hStop) );

	CUBLAS_CHECK( cublasDestroy(hCublas) );
	hCublas  = 0;
	CUDA_CHECK( cuCtxDestroy(hContext) );
	hContext = 0;

	// compare C and T for accuracy
	test(C, T, N, size);

	// And free up host memory
	free(A); free(B); free(C); free(T);

	return 0;
}

// Our kernel wrapper function
float assemblySgemm(CUdeviceptr devC, CUdeviceptr devA, CUdeviceptr devB, int N, CUevent hStart, CUevent hStop, int repeat, int printVars)
{
	// Configure our x and y grid dimensions (assume nice square matrixes).
	// Each block gets 128 tracks from A and 128 tracks from B.
	// Each of the 256 threads calculates 64 elements of that 128x128 sub matrix of C.
	// See Figure 2 here to get the gist of things (we use a different mapping to maximize LDS.128 usage):
	// http://icl.cs.utk.edu/projectsfiles/magma/pubs/fermi_gemm.pdf

	int gridDimXY = N/128 + (N%128 != 0);
	int blocks    = gridDimXY * gridDimXY;
	size_t size   = sizeof(float)*N*N;

	// Setup out debug printf output buffer
	CUdeviceptr devD = NULL; 
	int* D = NULL;
	int  sizeD = 0;

	if (printVars)
	{
		sizeD = blocks * 256 * printVars * sizeof(int);
		D     = (int*)malloc(sizeD);

		CUDA_CHECK( cuMemAlloc(&devD, sizeD) );
		CUDA_CHECK( cuMemsetD8(devD, 0, sizeD) );
	}

	// Load the cubin
	CUmodule hModule;
	CUDA_CHECK( cuModuleLoad(&hModule, "sgemm.cubin") );

	// Load the textures
	CUtexref texA, texB;
	CUDA_CHECK( cuModuleGetTexRef(&texA, hModule, "texA") );
	CUDA_CHECK( cuModuleGetTexRef(&texB, hModule, "texB") );

	// Configure the textures
	CUDA_CHECK( cuTexRefSetFormat(texA, CU_AD_FORMAT_FLOAT, 1) );
	CUDA_CHECK( cuTexRefSetFormat(texB, CU_AD_FORMAT_FLOAT, 1) );

	CUDA_CHECK( cuTexRefSetAddress(NULL, texA, devA, size) );
	CUDA_CHECK( cuTexRefSetAddress(NULL, texB, devB, size) );

	// Load the kernel function
	CUfunction hKernel;
	CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, "sgemm_kernel_128") );

	// Setup the params
	void* params[] = { &devC, &N, &N, &N, &N, &N, &N, &devD };

	CUDA_CHECK( cuEventRecord( hStart, NULL ) );
	
	// Launch the kernel
	for (int i = 0; i < repeat; i++)
		CUDA_CHECK( cuLaunchKernel(hKernel, gridDimXY, gridDimXY, 1, 256, 1, 1, 0, 0, params, 0) );

	float ms;
	CUDA_CHECK( cuEventRecord( hStop, NULL ) );
	CUDA_CHECK( cuEventSynchronize( hStop ) );
	CUDA_CHECK( cuEventElapsedTime( &ms, hStart, hStop ) );

	CUDA_CHECK( cuModuleUnload(hModule) );

	// And here we print out the debug info if requested:
	if (printVars)
	{
		CUDA_CHECK( cuMemcpyDtoH(D, devD, sizeD) );
		CUDA_CHECK( cuMemFree(devD) );
		float *fD = reinterpret_cast<float*>(D);

		for (int by = 0; by < gridDimXY; by++)
		{
			for (int bx = 0; bx < gridDimXY; bx++)
			{
				int blk = bx * by;
				for (int tid = 0; tid < 256; tid++)
				{
					int i = blk + tid;
					printf("by: %3d, bx: %3d, tid:%3d, rA:%5d, rB:%5d, wr:%5d, rd:%5d, cx:%5d, cy:%5d, ci:%5d, c:%.2f\n", 
						by, bx, tid, 
						D[i+0x000]/4,
						D[i+0x100]/4,
						D[i+0x200]/4,
						D[i+0x300]/4,
						D[i+0x400],
						D[i+0x500],
						D[i+0x600],
						fD[i+0x700]
					);
				}
			}
		}
		free(D);
	}

	return ms;
}

void gflops(const char* ident, int N, float ms, int repeat)
{
	// Standard sgemm flops formula
	ms /= repeat;
	printf("%s GFLOPS: %.2f (size: %d, iterations: %d)\n", ident, ((double)N * N * N * 2.0) / (ms * 1000000.0), N, repeat);
}

void test(float* C, float* T, int N, size_t size)
{
	// Compare our implementation with the cublas result
	int errors = memcmp(C, T, size);
	if (errors != 0)
	{
		if (N <= 768) // This gets too big and slow for large N
		{
			errors = 0;
			FILE* file;
			if (fopen_s(&file, "data.txt", "w") == 0)
			{
				for (int y = 0; y < N; ++y)
				{
					for (int x = 0; x < N; ++x)
					{
						float c = C[x*N + y];
						float t = T[x*N + y];
						if (c != t)
						{
							errors++;
							//printf("%d,%d: %.8f,%.8f\n", x,y, c, t);
							//fprintf(file, "%.0f!%.0f\t", c ,t, c);
							fprintf(file, "!\t");
						}
						else
						{
							//fprintf(file, "%.0f=%.0f\t", c ,t, c);
							fprintf(file, "=\t");
						}
					}
					fprintf(file, "\n");
				}
				fclose(file);
				printf("%d errors\n", errors);
			}
			else
				{ printf("Cannot open data.txt for writing\n"); }
		}
		else
			{ printf("%d errors\n", errors); }
	}
	else
		{ printf("%d errors\n", errors); }
}