
// Note this file isn't configured to automatically compile.
// Here's how:

// If you want to look at the ptx first:
// nvcc -arch sm_50 -m 32 -ptx sgemm.cu

// Manually compile your kernel to a cubin.
// You should only have to do this once, unless you change params or shared size or globals:
// nvcc -arch sm_50 -m 32 -cubin sgemm.cu

// If tweaking a kernel or writing a new one based on this shell code you would then do this:
// maxas.pl -e kernel.cubin kernel.sass

// I've already included a modified kernel (sgemm.sass) so the next step is..

// Splice the manually assembled code back into the cubin:
// maxas.pl -i sgemm.sass sgemm.cubin

#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>

typedef texture<float4, cudaTextureType1D, cudaReadModeElementType> floatTex;

floatTex  texA(0, cudaFilterModePoint, cudaAddressModeBorder);
floatTex  texB(0, cudaFilterModePoint, cudaAddressModeBorder);

// Use extern C so C++ doesn't mangle our kernel name
extern "C"
// This kernel requires 256x1x1 threads per block
__global__ void __launch_bounds__(256) sgemm_kernel_128(
	float *C,
	const int m,   const int n,   const int k,
	const int lda, const int ldb, const int ldc,
	float alpha, int *D)
{
	// Declare any shared memory your kernel requires
	// Or you could just pass the amount in as a param to cuLaunchKernel
	__shared__ float4 share[1024];

	int tid = threadIdx.x;

	// If you use indirect texture references, they will be passed as params at the end of the param list
	// So set that up here to make sure they're available in your kernel
	floatTex tex = tid > 127 ? texB : texA;

	// Make use of shared and your textures so it doesn't get optimized away
	share[tid] = tex1Dfetch(tex, tid);

	__syncthreads();

	// output something so your setup isn't optimized away.
	C[tid] = share[255-tid].x;
}

extern "C"
__global__ void __launch_bounds__(64) sgemm_kernel_64(
	float *C,
	const int m,   const int n,   const int k,
	const int lda, const int ldb, const int ldc,
	float alpha, int *D)
{
	__shared__ float4 share[512];

	int tid = threadIdx.x;

	floatTex tex = tid > 127 ? texB : texA;

	share[tid] = tex1Dfetch(tex, tid);

	__syncthreads();

	C[tid] = share[255-tid].x;
}

// A note about using the Cuda Runtime.
// If that's your preference over the driver API then here's what you'd do:

// In your project properties in the Cuda C/C++ panel:
//    -Set the "Keep Processed Files" (-keep) option
//    -Add a -v manually to the command line
// If compiling on command line just add -keep -v options to nvcc.
// Rebuild your solution and look in the log for these lines that follow the ptxas step:

// #$ fatbinary --create="Release/kernel.fatbin" -32 --key="a7bce87544c2a492" --ident="C:/Users/Scott/Documents/sgemm6/sgemm6/kernel.cu" --cmdline="-v --opt-level 4 --generate-line-info " "--image=profile=sm_50,file=Release/kernel.sm_50.cubin" "--image=profile=compute_50,file=Release/kernel.ptx" --embedded-fatbin="Release/kernel.fatbin.c" --cuda
// #$ cl.exe @Release/kernel.cu.cpp.ii.res > "Release/kernel.cu.cpp.ii"
// #$ cl.exe @Release/kernel.cu.obj.res -Fo"Release/kernel.cu.obj"

// You just need to manually run these 3 commands (or add them to a build script)
// after you've modified the cubin generated from the preceeding ptxas command.
// That will give you a new .cu.obj file which will automatically be linked in for you next time you
// build your project (or you could manually run the linker step as well).

// Having done that you can call your kernel normally using the <<< >>> syntax.
// Debugging will have to be with the sass syntax but that's what you'll want to see anyway.
// With fatbin you can also keep non-maxwell optimized versions of your code.


// I just discovered this also works as a shortcut to the above:
// nvcc -lib -arch sm_52 -m 32 -use-cubin code=sm_52,cubin=sgemm.cubin -o sgemm.lib sgemm.cu

// The cu kernel definitions above need to have empty bodies.
// And, the cu file must be compiled to a lib seperately before linking.