#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int *a, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]=a[i]*2+1; }
int main(){ int n=1<<20; int *a=nullptr; cudaError_t e=cudaMallocManaged(&a,n*sizeof(int)); if(e){printf("cudaMallocManaged FAIL %s\n",cudaGetErrorString(e));return 2;}
 for(int i=0;i<n;i++)a[i]=i; k<<<(n+255)/256,256>>>(a,n); e=cudaDeviceSynchronize(); if(e){printf("kernel FAIL %s\n",cudaGetErrorString(e));return 3;}
 long long s=0; for(int i=0;i<n;i++)s+=a[i]; printf("managed OK sum=%lld expect=%lld\n",s,(long long)n*(n-1)+n); 
 int dev=0; cudaDeviceProp p; cudaGetDeviceProperties(&p,dev); printf("dev %s cc %d.%d managedMemory=%d concurrentManagedAccess=%d pageableMemoryAccess=%d\n",p.name,p.major,p.minor,p.managedMemory,p.concurrentManagedAccess,p.pageableMemoryAccess); return 0; }
