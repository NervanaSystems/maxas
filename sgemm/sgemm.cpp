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

float assemblySgemm(const char* kernel, CUarray_format format, size_t size, CUdeviceptr devC, CUdeviceptr devA, CUdeviceptr devB, int N, CUevent hStart, CUevent hStop, int repeat = 1, int printVars = 0);
float cublasSgemm(const char* kernel, CUdeviceptr devC, CUdeviceptr devA, CUdeviceptr devB, int N, CUevent hStart, CUevent hStop, int repeat);
void gflops(const char* ident, int N, float ms, int repeat);
void test(float* C, float* T, int N, size_t size);

#define REPEAT_BLOCK 2000

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
	CUdeviceptr devA, devB, devC, devT, otherDevA, otherDevB;

	// Initialize the Driver API and find a device
	CUDA_CHECK( cuInit(0) );
	CUDA_CHECK( cuDeviceGetCount(&count) );
	for (ordinal = 0; ordinal < count; ordinal++)
	{
		CUDA_CHECK( cuDeviceGet(&hDevice, ordinal) );
		CUDA_CHECK( cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice) );
		CUDA_CHECK( cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice) );
		CUDA_CHECK( cuDeviceGetName(deviceName, sizeof(deviceName), hDevice) );
		if (major >= 5 && minor >= 2)
		{
			//printf("Using: Id:%d %s (%d.%d)\n\n", ordinal, deviceName, major, minor);
			break;
		}
	}
	if (ordinal == count)
	{
		printf("No compute 5.0 device found, exiting.\n");
		exit(EXIT_FAILURE);
	}

	// First command line arg is the size of N divided by 128
	int thread128 = 64;
	if (argc > 1)
		thread128 = atoi(argv[1]);
	if (thread128 > 64 || thread128 < 1)
		thread128 = 64;

	// Second command line arg is the repeat count for benchmarking
	int repeat = 1;
	if (argc > 2)
		repeat = atoi(argv[2]);
	if (repeat > 10000 || repeat < 1)
		repeat = 1;

	// Third command line arg is the normalized float size
	CUarray_format format = CU_AD_FORMAT_FLOAT;
	if (argc > 3)
		format = (CUarray_format)atoi(argv[3]);
	if (format != CU_AD_FORMAT_FLOAT && format != CU_AD_FORMAT_UNSIGNED_INT16 && format != CU_AD_FORMAT_UNSIGNED_INT8)
		format = CU_AD_FORMAT_FLOAT;

	// Forth command line arg is for printf debugging 
	int printVars = 0;
	if (argc > 4)
		printVars = atoi(argv[4]);
	if (printVars > 100 || printVars < 1)
		printVars = 0;

	int N = thread128 * 128;
	float alpha = 1, beta = 0, ms = 1;
	size_t sizeOther = N * N;
	size_t sizeFloat = sizeOther * 4;

	float* A = (float*)malloc(sizeFloat);
	float* B = (float*)malloc(sizeFloat);
	float* C = (float*)malloc(sizeFloat);
	float* T = (float*)malloc(sizeFloat);  
	float *otherA, *otherB; 

	//int counter = 0;
	//srand((unsigned int)time(0));
	for(int i = 0; i < N * N; i++) //
	{
		//A[i] = (float)rand() / (float)RAND_MAX;
		//B[i] = (float)rand() / (float)RAND_MAX;
		A[i] = B[i] = 1.0f; // * (i & 3) + 1.0f;
		//A[i] = 1.0f;
		//B[i * N + counter++] = 1.0f; // identity matrix
	}

	if (format == CU_AD_FORMAT_FLOAT)
	{
		sizeOther *= 4;
		otherA = A;
		otherB = B;
	}
	else if (format == CU_AD_FORMAT_UNSIGNED_INT16)
	{
		sizeOther *= 2;
		unsigned short* othera = (unsigned short*)malloc(sizeOther);
		unsigned short* otherb = (unsigned short*)malloc(sizeOther);
		for(int i = 0; i < N * N; i++)
			othera[i] = otherb[i] = 65535;

		otherA = reinterpret_cast<float*>(othera);
		otherB = reinterpret_cast<float*>(otherb);
	}
	else // (format == CU_AD_FORMAT_UNSIGNED_INT8)
	{
		otherA = (float*)malloc(sizeOther);
		otherB = (float*)malloc(sizeOther);
		memset(otherA, 255, sizeOther);
		memset(otherB, 255, sizeOther); 
	}

	CUDA_CHECK( cuCtxCreate(&hContext, 0, hDevice) );
	//CUBLAS_CHECK( cublasCreate(&hCublas) );
	
	CUDA_CHECK( cuEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC) ); // CU_EVENT_DEFAULT 
	CUDA_CHECK( cuEventCreate(&hStop,  CU_EVENT_BLOCKING_SYNC) );

	CUDA_CHECK( cuMemAlloc(&devA, sizeFloat) );
	CUDA_CHECK( cuMemAlloc(&devB, sizeFloat) );
	CUDA_CHECK( cuMemAlloc(&devC, sizeFloat) );
	CUDA_CHECK( cuMemAlloc(&devT, sizeFloat) );
	
	CUDA_CHECK( cuMemcpyHtoD(devA, A, sizeFloat) );
	CUDA_CHECK( cuMemcpyHtoD(devB, B, sizeFloat) );
	CUDA_CHECK( cuMemsetD8(devC, 0, sizeFloat) );
	CUDA_CHECK( cuMemsetD8(devT, 0, sizeFloat) );

	if (format == CU_AD_FORMAT_FLOAT)
	{
		otherDevA = devA;
		otherDevB = devB;
	}
	else
	{
		CUDA_CHECK( cuMemAlloc(&otherDevA, sizeOther) );
		CUDA_CHECK( cuMemAlloc(&otherDevB, sizeOther) );
		CUDA_CHECK( cuMemcpyHtoD(otherDevA, otherA, sizeOther) );
		CUDA_CHECK( cuMemcpyHtoD(otherDevB, otherB, sizeOther) );
	}

	// Warm up the clock (unless under nsight)
	//if (!getenv("NSIGHT_LAUNCHED")) // NSIGHT_CUDA_ANALYSIS NSIGHT_CUDA_DEBUGGER 
	//	for (int i = 0; i < 3; i++)
	//		CUBLAS_CHECK( cublasSgemm(hCublas, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, reinterpret_cast<float*>(devA), N, reinterpret_cast<float*>(devB), N, &beta, reinterpret_cast<float*>(devT), N) );

	// Launch our kernel
	ms = assemblySgemm("sgemm_kernel_64", format, sizeOther, devC, otherDevA, otherDevB, N, hStart, hStop, repeat, printVars);
	gflops("Max64 ", N, ms, repeat);

	ms = assemblySgemm("sgemm_kernel_128", format, sizeOther, devC, otherDevA, otherDevB, N, hStart, hStop, repeat, printVars);
	gflops("Max128", N, ms, repeat);

	//ms = cublasSgemm("maxwell_sgemm_128x64_nt", devT, devA, devB, N, hStart, hStop, repeat);
	//gflops("Cub64 ", N, ms, repeat);

	//ms = cublasSgemm("maxwell_sgemm_128x128_nt", devT, devA, devB, N, hStart, hStop, repeat);
	//gflops("Cub128", N, ms, repeat);

	// Run cublas again for the same repeat count for comparison
	//CUDA_CHECK( cuEventRecord(hStart, NULL) );
	//for (int i = 0; i < repeat; i++)
	//	CUBLAS_CHECK( cublasSgemm(hCublas, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, reinterpret_cast<float*>(devA), N, reinterpret_cast<float*>(devB), N, &beta, reinterpret_cast<float*>(devT), N) );
	//CUDA_CHECK( cuEventRecord(hStop, NULL) );
	//CUDA_CHECK( cuEventSynchronize(hStop) );
	//CUDA_CHECK( cuEventElapsedTime(&ms, hStart, hStop) );
	//gflops("Cublas", N, ms, repeat);

	// Get back our results from each kernel
	CUDA_CHECK( cuMemcpyDtoH(C, devC, sizeFloat) );
	CUDA_CHECK( cuMemcpyDtoH(T, devT, sizeFloat) );
	
	// Cleanup and shutdown of cuda
	CUDA_CHECK( cuMemFree(devA) );
	CUDA_CHECK( cuMemFree(devB) );
	CUDA_CHECK( cuMemFree(devC) );
	CUDA_CHECK( cuMemFree(devT) );
	if (format != CU_AD_FORMAT_FLOAT)
	{
		CUDA_CHECK( cuMemFree(otherDevA) );
		CUDA_CHECK( cuMemFree(otherDevB) );
	}

	CUDA_CHECK( cuEventDestroy(hStart) );
	CUDA_CHECK( cuEventDestroy(hStop) );

	//CUBLAS_CHECK( cublasDestroy(hCublas) );
	//hCublas  = 0;
	CUDA_CHECK( cuCtxDestroy(hContext) );
	hContext = 0;

	// compare C and T for accuracy
	test(C, T, N, sizeFloat);

	// And free up host memory
	free(A); free(B); free(C); free(T);

	if (format != CU_AD_FORMAT_FLOAT)
	{
		free(otherA); 
		free(otherB);
	}

	return 0;
}

// Our kernel wrapper function
float assemblySgemm(const char* kernel, CUarray_format format, size_t size, CUdeviceptr devC, CUdeviceptr devA, CUdeviceptr devB, int N, CUevent hStart, CUevent hStop, int repeat, int printVars)
{
	// Configure our x and y grid dimensions (assume nice square matrixes).
	// Each block gets 128 tracks from A and 128 tracks from B.
	// Each of the 256 threads calculates 64 elements of that 128x128 sub matrix of C.
	// See Figure 2 here to get the gist of things (we use a different mapping to maximize LDS.128 usage):
	// http://icl.cs.utk.edu/projectsfiles/magma/pubs/fermi_gemm.pdf

	int threads, width;
	if (strcmp(kernel, "sgemm_kernel_64") == 0)
	{
		threads = 64;
		width   = 64;
	}
	else
	{
		threads = 256;
		width   = 128;
	}

	int gridDimXY = N / width + (N % width != 0);
	int blocks    = gridDimXY * gridDimXY;

	// Setup out debug printf output buffer
	CUdeviceptr devD = NULL; 
	int* D = NULL;
	int  sizeD = 0;

	if (printVars)
	{
		sizeD = blocks * threads * printVars * sizeof(int);
		D = (int*)malloc(sizeD);

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
	CUDA_CHECK( cuTexRefSetFormat(texA, format, 4) );
	CUDA_CHECK( cuTexRefSetFormat(texB, format, 4) );

	CUDA_CHECK( cuTexRefSetAddress(NULL, texA, devA, size) );
	CUDA_CHECK( cuTexRefSetAddress(NULL, texB, devB, size) );

	// Load the kernel function
	CUfunction hKernel;
	CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, kernel) );

	// Setup the params
	float alpha = 1.0f;
	void* params[] = { &devC, &N, &N, &N, &N, &N, &N, &alpha, &devD };

	float totalTime = 0;
	// Launch the kernel repeat times.. but break it up into pieces so as not to lock things up.
	while (repeat > 0)
	{
		float ms;
		int r = repeat > REPEAT_BLOCK ? REPEAT_BLOCK : repeat;
		CUDA_CHECK( cuEventRecord( hStart, NULL ) );
		
		for (int i = 0; i < r; i++)
			CUDA_CHECK( cuLaunchKernel(hKernel, gridDimXY, gridDimXY, 1, threads, 1, 1, 0, 0, params, 0) );
		
		CUDA_CHECK( cuEventRecord( hStop, NULL ) );
		CUDA_CHECK( cuEventSynchronize( hStop ) );
		CUDA_CHECK( cuEventElapsedTime( &ms, hStart, hStop ) );
		totalTime += ms;
		repeat -= r;
	}


	CUDA_CHECK( cuModuleUnload(hModule) );

	// And here we print out the debug info if requested:
	if (printVars)
	{
		CUDA_CHECK( cuMemcpyDtoH(D, devD, sizeD) );
		CUDA_CHECK( cuMemFree(devD) );
		int   *iD = D;
		float *fD = reinterpret_cast<float*>(D);
		unsigned int *uD = reinterpret_cast<unsigned int*>(D);

		for (int by = 0; by < gridDimXY; by++)
		{
			for (int bx = 0; bx < gridDimXY; bx++)
			{
				unsigned int clock = 0xffffffff, sm = 0;

				for (int tid = 0; tid < threads; tid++)
				{
					//printf("by: %3d, bx: %3d, tid:%3d, rA:%5d, rB:%5d, wr:%5d, rd:%5d, cx:%5d, cy:%5d, ci:%5d, c:%.2f\n", 
					//printf("by: %3d, bx: %3d, tid:%3d, t0:%5d, end:%5d, k:%5d, tid2:%5d, tid15:%5d, ldx:%5d, t2:%5d, t4:%5d\n", 
					//	    by,      bx,      tid,     iD[0],  iD[1],   iD[2], iD[3],    iD[4],     iD[5],   iD[6],  iD[7]
					//);
					if (uD[1] < clock) clock = uD[1];
					sm = uD[0];

					iD += printVars;
					fD += printVars;
					uD += printVars;
				}
				printf("%02d %08u %d %d\n", sm, clock, by, bx);
			}
		}
		free(D);
	}

	return totalTime;
}

typedef struct dPointer
{
	CUdeviceptr lo;
	CUdeviceptr hi;
} dPointer;

float cublasSgemm(const char* kernel, CUdeviceptr devC, CUdeviceptr devA, CUdeviceptr devB, int N, CUevent hStart, CUevent hStop, int repeat)
{
	int threads, gridX, gridY;
	if (strcmp(kernel, "maxwell_sgemm_128x64_nt") == 0)
	{
		threads = 128;
		gridX = N / 128 + (N % 128 != 0);
		gridY = N / 64  + (N % 64  != 0);
	}
	else
	{
		threads = 256;
		gridX = gridY = N / 128 + (N % 128 != 0);
	}
	int blocks = gridX * gridY;

	// Load the cubin
	// See cublas_sgemm.ptx for info on how to build this.
	CUmodule hModule;
	CUDA_CHECK( cuModuleLoad(&hModule, "cublas_sgemm.cubin") );

	// Load the kernel function
	CUfunction hKernel;
	CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, kernel) );

	// Setup the params
	// I should probably be working in 64 bits...
	dPointer dA = { devA, 0 };
	dPointer dB = { devB, 0 };
	dPointer dC = { devC, 0 };

	int   flag  = 0;
	float alpha = 1.0;
	float beta  = 0.0;
	
	void* params[] = { &dA, &dB, &dC, &N, &N, &N, &N, &dA, &dA, &alpha, &beta, &flag };

	float totalTime = 0;
	// Launch the kernel repeat times.. but break it up into pieces so as not to lock things up.
	while (repeat > 0)
	{
		float ms;
		int r = repeat > REPEAT_BLOCK ? REPEAT_BLOCK : repeat;
		CUDA_CHECK( cuEventRecord( hStart, NULL ) );
		
		for (int i = 0; i < r; i++)
			CUDA_CHECK( cuLaunchKernel(hKernel, gridX, gridY, 1, threads, 1, 1, 0, 0, params, 0) );
		
		CUDA_CHECK( cuEventRecord( hStop, NULL ) );
		CUDA_CHECK( cuEventSynchronize( hStop ) );
		CUDA_CHECK( cuEventElapsedTime( &ms, hStart, hStop ) );
		totalTime += ms;
		repeat -= r;
	}


	CUDA_CHECK( cuModuleUnload(hModule) );

	return totalTime;
}

void gflops(const char* ident, int N, float ms, int repeat)
{
	// Standard sgemm flops formula
	ms /= repeat;
	printf("%s GFLOPS: %.2f (size: %d, iterations: %d)\n", ident, ((double)N * N * N * 2.0 + N * N) / (ms * 1000000.0), N, repeat);
}

void test(float* C, float* T, int N, size_t size)
{
	// Compare our implementation with the cublas result
	int errors = memcmp(C, T, size);
	if (errors)
	{
		if (N <= 512) // This gets too big and slow for large N
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
							fprintf(file, "%.8f!%.8f\t", c , t);
							//fprintf(file, "%.0f!", c);
							//fprintf(file, "!");
						}
						else
						{
							//fprintf(file, "%.0f=%.0f\t", c , t);
							//fprintf(file, "%.0f=", c);
							fprintf(file, "=");
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