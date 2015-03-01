#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <conio.h>

#define _USE_MATH_DEFINES
#include "math.h"


//#define N 16
//#define N 4194304
#define N 16777216



float data_real[N];
float data_imag[N];
int reverse[N];
__global__ void ppt2(float* data_real_d,float* data_imag_d)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int x=tid;
	unsigned int slash=8;
	float tmp;
	int offset=N/slash;

	unsigned int index=x;
	unsigned h = 0;
    int i;
     // loop through all the bits
    for(i = 0; i < __log2f(N); i++)
    {
          // add bit from value to 1 bit left shifted variable
        h = (h << 1) + (x & 1);
        // right shift bits by 1
        x >>= 1;
    }
	if(h>index)
	{tmp=data_real_d[index];
	data_real_d[index]=data_real_d[h];
	data_real_d[h]=tmp;

	tmp=data_imag_d[index];
	data_imag_d[index]=data_imag_d[h];
	data_imag_d[h]=tmp;}
	
	for(int j=1;j<slash;j++)
	{x=tid+offset*j;
	index=x;
	h = 0;
    
     // loop through all the bits
    for(i = 0; i < __log2f(N); i++)
    {
          // add bit from value to 1 bit left shifted variable
        h = (h << 1) + (x & 1);
        // right shift bits by 1
        x >>= 1;
    }

	if(h>index)
	{tmp=data_real_d[index];
	data_real_d[index]=data_real_d[h];
	data_real_d[h]=tmp;

	tmp=data_imag_d[index];
	data_imag_d[index]=data_imag_d[h];
	data_imag_d[h]=tmp;}
	}

}


__global__ void ppt(float* data_real_d,float* data_imag_d,int* reverse_d)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	float tmp;
	int value;
	int slash=2;
	int offset=N/slash;
	for(int i=0;i<slash;i++)
	{	tid=tid+offset*i;
		value=reverse_d[tid];
		if(value>tid)
		{
		tmp=data_imag_d[tid];
		data_imag_d[tid]=data_imag_d[value];
		data_imag_d[value]=tmp;
	
		tmp=data_real_d[tid];
		data_real_d[tid]=data_real_d[value];
		data_real_d[value]=tmp;
		}
	}

}

__global__ void fft(float* data_real_d,float* data_imag_d,int p)
{	
	
	unsigned int x,block,sub,index;	
	float tw_real;
	float tw_imag;
	unsigned int power;
	float tmp;
	float real,real2,imag,imag2;
	index=threadIdx.x+blockIdx.x*blockDim.x;
		
		power=__powf(2,p);
		//determine which block the thread is in(not cuda block)
		//x=N/(power*2);
		x=N>>(p+1);
		//block=(index)/x;
		//tmp2=__log2f(x);
		block=index>>(int)(__log2f(x));
		//sub is the subscript of the array where the thread should get his element1 for processing
		//tmp2=__log2f(block);
		//tmp2=x<<tmp2;
		sub=index+x*block;
		//issue request for real parts
		 real=data_real_d[sub];
		 real2=data_real_d[sub+x];
		//compute twiddle factor
		//tmp=(index)%x;
		tmp=(index)&(x-1);
		tmp=(2*M_PI*tmp*power)/N;
		tw_real=cosf(tmp);
		tw_imag=-1*sinf(tmp);
		//issue request for imaginary parts
		imag=data_imag_d[sub];
		imag2=data_imag_d[sub+x];
		//butterfly real parts
		tmp=real+real2;
		real2=real-real2;
		real=tmp;
		//write back real results of butterfly,only this part is written because we still need to twiddle the other
		data_real_d[sub]=real;
		//butterfly imag part
		tmp=imag+imag2;
		imag2=imag-imag2;
		imag=tmp;
		//multiply by twiddle
		tmp=real2;
		real2=real2*tw_real-imag2*tw_imag;
		data_real_d[sub+x]=real2;
		imag2=tmp*tw_imag+imag2*tw_real;
		//write back imag result of butterfly
		data_imag_d[sub]=imag;
		data_imag_d[sub+x]=imag2;
}



void bit_reversal()
{
   long i,i1,j,k,i2;
   double c1,c2,tx,ty;
   i2 = N >> 1;
   j = 0;
   for (i=0;i<N-1;i++) {
      if (i < j) {
         tx = data_real[i];
         ty = data_imag[i];
         data_real[i] = data_real[j];
         data_imag[i] = data_imag[j];
         data_real[j] = tx;
         data_imag[j] = ty;
      }
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }
}


int main( int argc, char** argv) 
{
	for(int i=0;i<N;i++)
	{	
		if(i<N/2) 
		{data_real[i]=1;
		data_imag[i]=0;}
		else{
		data_real[i]=0;
		data_imag[i]=0;
		}
	}
	
unsigned int h,index;
	for(int j=0;j<N;j++)
	{
	
	index=j;
	h = 0;
    
     // loop through all the bits
    for(int i = 0; i < log2f(N); i++)
    {
          // add bit from value to 1 bit left shifted variable
        h = (h << 1) + (index & 1);
        // right shift bits by 1
        index >>= 1;
    }
	//store value of h
	reverse[j]=h;
	}
printf("reverse[0]=%d\n",reverse[01]);

	int passes=log((float)N)/log((float)2);
	float* data_real_d;
	float* data_imag_d;
	//int* reverse_d;
	cudaMalloc((void**)&data_real_d,N*sizeof(float));
	cudaMalloc((void**)&data_imag_d,N*sizeof(float));
	//cudaMalloc((void**)&reverse_d,N*sizeof(int));
	cudaMemcpy(data_real_d,data_real,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(data_imag_d,data_imag,sizeof(float)*N,cudaMemcpyHostToDevice);
	//cudaMemcpy(reverse_d,reverse,sizeof(int)*N,cudaMemcpyHostToDevice);

	dim3 dimBlock(512,1,1);
	dim3 dimGrid(N/1024,1,1);
	cudaThreadSynchronize();
	long int before = GetTickCount();
//-----------------------	
	cudaEvent_t start, stop; float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop); cudaEventRecord( start, 0 );
	for(int i=0;i<passes;i++)
	{
		fft<<<dimGrid,dimBlock>>>(data_real_d,data_imag_d,i);
	}
	cudaThreadSynchronize();
	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
printf("fft time=%f\n",time);
//------------------------
	cudaEventCreate(&start);
	cudaEventCreate(&stop); cudaEventRecord( start, 0 );
//    ppt<<<dim3(N/1024,1,1),dimBlock>>>(data_real_d,data_imag_d,reverse_d);
	ppt2<<<dim3(N/4096,1),dimBlock>>>(data_real_d,data_imag_d);
	cudaThreadSynchronize();
	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
//-----------------
	long int after = GetTickCount();

	const char* err=cudaGetErrorString(cudaGetLastError());
	
	for(int i=0;i<40;i++)
	{printf("%c",err[i]);}
	printf("\n");
	printf("%d ms\n",after-before);


	cudaMemcpy(data_real,data_real_d,4*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(data_imag,data_imag_d,4*N,cudaMemcpyDeviceToHost);
	cudaFree(data_real_d);
	cudaFree(data_imag_d);
	long int before2 = GetTickCount();
	//bit_reversal();	
	long int after2=GetTickCount();

for(int i=N-16;i<N;i++)
	{
		printf("data[%d]=%f + %f i\n",i,data_real[i],data_imag[i]);
	}


	printf("ppt time= %f ms\n",time);
	
		
//	_getch();



}

