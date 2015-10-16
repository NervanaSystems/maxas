
// Note this file isn't configured to automatically compile

#include <device_functions.h>
#include <device_launch_parameters.h>

// Build:
// nvcc -l cuda -o microbench microbench.cpp
// nvcc -arch sm_50 -cubin microbench.cu

// Inspect a cubin (use nvdisasm from cuda 6.5 for best results):
// maxas.pl -e microbench.cubin

// Insert new sass into cubin
// maxas.pl -i microbench.sass microbench.cubin

// run it:
// ./microbench

// Use extern C so C++ doesn't mangle our kernel name
extern "C" __global__ void  microbench(int *out, int *clocks, int *in)
{
    __shared__ int share[1024];

    int tid = threadIdx.x;
    int bx  = blockIdx.x;
    int by  = blockIdx.y;

    int start = clock();

    share[tid] = in[by * 65535 + bx]; //tid + blkDimX + blkDimY + blkDimZ + grdDimX + grdDimY + grdDimZ

    __syncthreads();

    int end = clock();

    clocks[tid] = (start >> 16) | (end & 0xffff0000); //end - start;

    out[tid] = share[tid ^ 1];
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
// nvcc -lib -arch sm_52 -m 32 -use-cubin code=sm_52,cubin=microbench.cubin -o microbench.lib microbench.cu

// The cu kernel definitions above need to have empty bodies.
// And, the cu file must be compiled to a lib seperately before linking.