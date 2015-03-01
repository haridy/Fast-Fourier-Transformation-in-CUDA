#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <conio.h>

//#define _USE_MATH_DEFINES
#include "math.h"
#define M_PI 3.141592653589793f
#define IT 1

//#define N 1024
//#define N 2048
//#define N 4096
//#define N 8192
//#define N 16384
//#define N 32768
//#define N 65536
//#define N 131072
//#define N 262144
//#define N 524288
//#define N 1048576
//#define N 2097152
//#define N 4194304
//#define N 8388608
//#define N 16777216
#define N 33554432
//#define N 67108864

float data_real[N];
float data_imag[N];




__global__ void stfft(float* data_real_d_in,float* data_imag_d_in,float* data_real_d_out,float* data_imag_d_out,int p)
{	
	
	int subarray1,subarray2,m,thread_position,subarray_start,subarray2_start,tmp2,tmp3;
	float tw_real;
	float tw_imag;
	int power;
	float tmp;
	float real,real2,imag,imag2;
	int	index=threadIdx.x+blockIdx.x*blockDim.x;

		//power=__powf(2,p);
		power = 1<<p;
		subarray1=index>>p;
		m=N>>(p+1);
		subarray2=subarray1+m;//7
		//thread_position=index%power;
		thread_position=(index)&(power-1);
		subarray_start=subarray1<<p;
		subarray2_start=subarray2<<p;
		tmp3=subarray_start+thread_position;
		tmp2=subarray2_start+thread_position;
		//issue request for real parts
		 real=data_real_d_in[tmp3];
		 real2=data_real_d_in[tmp2];//15
		//compute twiddle factor
		tmp=(index)&(m-1);//17
		tmp=(2*M_PI*subarray1*power)/N;
		//tw_real=cosf(tmp);
		//tw_imag=-1*sinf(tmp);
		sincosf(tmp,&tw_imag,&tw_real);
		tw_imag=tw_imag*-1;
		//issue request for imaginary parts
		imag=data_imag_d_in[tmp3];
		imag2=data_imag_d_in[tmp2];//19
		//butterfly real parts
		tmp=real+real2;
		real2=real-real2;
		real=tmp;
		//write back real results of butterfly,only this part is written because we still need to twiddle the other
		tmp2=subarray_start*2+thread_position;
		data_real_d_out[tmp2]=real;//22
		//butterfly imag part
		tmp=imag+imag2;
		imag2=imag-imag2;
		imag=tmp;
		//multiply by twiddle
		tmp=real2;
		real2=real2*tw_real-imag2*tw_imag;
		data_real_d_out[tmp2+power]=real2;
		imag2=tmp*tw_imag+imag2*tw_real;//10
		//write back imag result of butterfly
		data_imag_d_out[tmp2]=imag;
		data_imag_d_out[tmp2+power]=imag2;//27
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

	int passes=log((float)N)/log((float)2);
	int* cycles=(int*)malloc(N/2*sizeof(int));
	int* cycles_d;
	float* data_real_d;
	float* data_imag_d;
	float* data_real_d_out;
	float* data_imag_d_out;
	float* tmp;float* tmp2;
	float* fft_time=(float*)calloc(IT,sizeof(float));
	cudaEvent_t start, stop; float time;
	cudaMalloc((void**)&cycles_d,N/2*sizeof(int));
	cudaMalloc((void**)&data_real_d,N*sizeof(float));
	cudaMalloc((void**)&data_imag_d,N*sizeof(float));
	cudaMalloc((void**)&data_real_d_out,N*sizeof(float));
	cudaMalloc((void**)&data_imag_d_out,N*sizeof(float));
	dim3 dimBlock(512,1,1);
	dim3 dimGrid(N/1024,1,1);

	long int before = GetTickCount();

	

	//cudaFuncSetCacheConfig(stfft,cudaFuncCachePreferShared);
	//-----------------------	
	

for(int j=0;j<IT;j++)
{


	cudaMemcpy(data_real_d,data_real,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(data_imag_d,data_imag,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaEventCreate(&stop); 
	cudaEventCreate(&start);
	cudaEventRecord( start, 0 );
	for(int i=0;i<passes;i++)
	{
		
		stfft<<<dimGrid,dimBlock>>>(data_real_d,data_imag_d,data_real_d_out,data_imag_d_out,i);
		tmp=data_real_d;
		tmp2=data_imag_d;
		data_real_d=data_real_d_out;
		data_real_d_out=tmp;
		data_imag_d=data_imag_d_out;
		data_imag_d_out=tmp2;
	}
	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("fft time=%f\n",time);
	fft_time[j]=time;

		tmp=data_real_d;
		tmp2=data_imag_d;
		data_real_d=data_real_d_out;
		data_real_d_out=tmp;
		data_imag_d=data_imag_d_out;
		data_imag_d_out=tmp2;
}

		tmp=data_real_d;
		tmp2=data_imag_d;
		data_real_d=data_real_d_out;
		data_real_d_out=tmp;
		data_imag_d=data_imag_d_out;
		data_imag_d_out=tmp2;

	long int after = GetTickCount();
	const char* err=cudaGetErrorString(cudaGetLastError());	
	for(int i=0;i<40;i++)
	{printf("%c",err[i]);}
	printf("\n");
	printf("%d ms\n",after-before);

	cudaMemcpy(data_real,data_real_d,sizeof(float)*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(data_imag,data_imag_d,sizeof(float)*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(cycles,cycles_d,sizeof(int)*N/2,cudaMemcpyDeviceToHost);
	cudaFree(data_real_d);
	cudaFree(data_imag_d);
	
	for(int i=N-16;i<N;i++)
	{
		printf("data[%d]=%f + %f i\n",i,data_real[i],data_imag[i]);
	}
	
	float average=0;
	for(int i=0;i<IT;i++)
	{
		average+=fft_time[i];
	}
	average=average/IT;
	float flops=(41*(N/2)*log2f(N))/(average*0.001);
	printf("FLOPS=%f GFLOPS, AV Time=%f\n",flops*0.000000001,average);
/*
	for(int i=0;i<128;i++)
	{
		printf("cycles[%d]=%d\n",i,cycles[i]);
	}*/
	//_getch();



}

