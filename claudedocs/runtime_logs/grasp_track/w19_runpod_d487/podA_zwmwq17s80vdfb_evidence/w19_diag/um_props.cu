#include <cstdio>
#include <cuda_runtime.h>
int main(){ cudaDeviceProp p; cudaGetDeviceProperties(&p,0); int drv=0,rt=0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
printf("pageableMemoryAccessUsesHostPageTables=%d driverAPI=%d runtime=%d\n",p.pageableMemoryAccessUsesHostPageTables,drv,rt); return 0; }
