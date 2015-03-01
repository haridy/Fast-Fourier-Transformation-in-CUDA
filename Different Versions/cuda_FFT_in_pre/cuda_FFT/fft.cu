
#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#define _USE_MATH_DEFINES
#include "math.h"



//#define N 2048
#define N 8388608

float data_real[N];
float data_imag[N];
float tw_real[N/2];
float tw_imag[N/2];




__global__ void fft(float* data_real_d,float* data_imag_d,float* tw_real_d,float* tw_imag_d,int p)
{	
	
	unsigned int x,block,sub,index,tmp2;	
	float tw_real_reg;
	float tw_imag_reg;
	unsigned int power;
	float tmp;
	float real,real2,imag,imag2;
	index=threadIdx.x+blockIdx.x*blockDim.x;
		
		power=__powf(2,p);
		//determine which block the thread is in(not cuda block)
		//x=N/(power*2);
		//block=(index)/x;
		x=N>>(p+1);
		tmp2=__log2f(x);
		block=index>>tmp2;
		//sub is the subscript of the array where the thread should get his element1 for processing
		sub=index+(x*block);
		//issue request for real parts
		 real=data_real_d[sub];
		 real2=data_real_d[sub+x];
		
		//fetch twiddle factor
		
		//tmp=(index)%x;
		 tmp=(index)&(x-1);
		tw_real_reg=tw_real_d[(int)tmp*power];
		tw_imag_reg=tw_imag_d[(int)tmp*power];
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
		real2=real2*tw_real_reg-imag2*tw_imag_reg;
		data_real_d[sub+x]=real2;
		imag2=tmp*tw_imag_reg+imag2*tw_real_reg;
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


void compute_twiddle()
{
	for(int i=0;i<N/2;i++)
	{
		tw_real[i]=cos(2*M_PI*i/N);
		tw_imag[i]=-sin(2*M_PI*i/N);
		
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

//printf("data[0]=%f + %f i\n",data_real[0],data_imag[0]);

///	for(int i=0;i<N;i++)
//	{
//		printf("data[%d]=%f + %f i\n",i,data_real[i],data_imag[i]);
//	}


	compute_twiddle();
	int passes=log((float)N)/log((float)2);
	float* data_real_d;
	float* data_imag_d;
	float* tw_real_d;
	float* tw_imag_d;
	
	cudaMalloc((void**)&data_real_d,N*sizeof(float));
	cudaMalloc((void**)&data_imag_d,N*sizeof(float));
	cudaMalloc((void**)&tw_imag_d,(N/2)*sizeof(float));
	cudaMalloc((void**)&tw_real_d,(N/2)*sizeof(float));
	
	cudaMemcpy(data_real_d,data_real,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(data_imag_d,data_imag,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(tw_real_d,tw_real,sizeof(float)*(N/2),cudaMemcpyHostToDevice);
	cudaMemcpy(tw_imag_d,tw_imag,sizeof(float)*(N/2),cudaMemcpyHostToDevice);
	dim3 dimBlock(512,1,1);
	dim3 dimGrid(N/1024,1,1);
cudaThreadSynchronize();
	long int before = GetTickCount();
	cudaEvent_t start, stop; float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop); cudaEventRecord( start, 0 );

	for(int i=0;i<passes;i++)
	{fft<<<dimGrid,dimBlock>>>(data_real_d,data_imag_d,tw_real_d,tw_imag_d,i);}

cudaThreadSynchronize();
	long int after = GetTickCount();
	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	const char* err=cudaGetErrorString(cudaGetLastError());
	
	for(int i=0;i<10;i++)
	{printf("%c",err[i]);}
	printf("\n");
	printf("%d ms\n",after-before);


	cudaMemcpy(data_real,data_real_d,4*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(data_imag,data_imag_d,4*N,cudaMemcpyDeviceToHost);
	
	cudaFree(data_real_d);
	cudaFree(data_imag_d);
	cudaFree(tw_real_d);
	cudaFree(tw_imag_d);
	
	bit_reversal();


for(int i=N-10;i<N;i++)
	{
		printf("data[%d]=%f + %f i\n",i,data_real[i],data_imag[i]);
	}
//printf("data[0]=%f + %f i\n",data_real[0],data_imag[0]);
//printf("data[1]=%f + %f i\n",data_real[21],data_imag[21]);

	printf("cuda timer record %f ms",time);
	

	scanf("%d");



}

