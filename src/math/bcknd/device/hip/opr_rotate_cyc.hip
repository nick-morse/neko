#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <device/device_config.h>
#include <device/hip/check.h>
#include "rotate_kernel.h"



extern "C" {

void hip_rotate_cyc(void *rx, void *ry, void *rz,
                    void *x, void *y, void *z,
                    void *cyclic_angle,
                    int *idir, int *nel, int *lx){  
    
    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks((*nel), 1, 1);
    
#define CASE(LX)                                                                        \
    case LX:                                                                            \
        hipLaunchKernelGGL(HIP_KERNEL_NAME( rotate_cyc_kernel<real, LX, 1024> ), \
                           nblcks, nthrds, 0, (hipStream_t) glb_cmd_queue,      \
                           (real *) rx, (real *) ry, (real *) rz,                     \
                           (real *) x,  (real *) y,  (real *) z,                       \
                           (real *) cyclic_angle, *idir);                             \
        HIP_CHECK(hipGetLastError());                                               \
        break

    switch(*lx) {
        CASE(2);
        CASE(3);
        CASE(4);
        CASE(5);
        CASE(6);
        CASE(7);
        CASE(8);
        CASE(9);
        CASE(10);
    default:
        {
            fprintf(stderr, __FILE__ ": size not supported: %d\n", *lx);
            exit(1);
        }
    }
}
}


