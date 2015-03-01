#include <CL/cl.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <windows.h>

#define _USE_MATH_DEFINES
#include "math.h"
#define N 33554432
	
float data_real[N];
float data_imag[N];
int 
main(int argc, char * argv[])
{

//creating input data

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
//number of passes
	int passes=log((float)N)/log((float)2);;
	


	cl_int clerr=CL_SUCCESS;
//create context and command queue

// Get platforms
cl_uint nPlatform = 0;
clGetPlatformIDs(0,NULL,&nPlatform);
cl_platform_id* plInfos = (cl_platform_id*)malloc(nPlatform * sizeof(cl_platform_id));
clGetPlatformIDs(nPlatform,plInfos,NULL);
// Get Devices
cl_uint nDev = 0;
clGetDeviceIDs(plInfos[0],CL_DEVICE_TYPE_GPU,0,0,&nDev);
cl_device_id* GPUDevices = (cl_device_id*)malloc(nDev * sizeof(cl_device_id));
clGetDeviceIDs(plInfos[0],CL_DEVICE_TYPE_GPU,nDev,GPUDevices,0);
cl_context clctx = clCreateContext(0, 1, GPUDevices, NULL, NULL, &clerr);

//cl_context clctx=clCreateContextFromType(0,CL_DEVICE_TYPE_ALL,NULL,NULL,&clerr);
//	printf("Test success till here\n");
	size_t parmsz;
	clerr=clGetContextInfo(clctx,CL_CONTEXT_DEVICES,0,NULL,&parmsz);
	cl_device_id* cldevs=(cl_device_id*)malloc(parmsz);
	clerr=clGetContextInfo(clctx,CL_CONTEXT_DEVICES,parmsz,cldevs,NULL);
	cl_command_queue clcmdq=clCreateCommandQueue(clctx,cldevs[0],0,&clerr);

//the kernel represented as a string
	const char* kernel="__kernel void stfft(__global float* data_real_d_in,__global float* data_imag_d_in,__global float* data_real_d_out,__global float* data_imag_d_out,const int p, const unsigned int N)\n"
"{\n"
	"unsigned int subarray1,subarray2,m,thread_position,subarray_start,subarray2_start,index,power;\n"
	"float real,real2,imag,imag2,tmp,tw_real,tw_imag;\n"
		"index=get_global_id(0);\n"
		"power = 1<<p;\n"
		"subarray1=index>>p;\n"
		"m=N>>(p+1);\n"
		"subarray2=subarray1+m;\n"
		"thread_position=(index)&(power-1);\n"
		"subarray_start=subarray1<<p;\n"
		"subarray2_start=subarray2<<p;\n"
		 "real=data_real_d_in[subarray_start+thread_position];\n"
		 "real2=data_real_d_in[subarray2_start+thread_position];\n"
		"tmp=(2*3.14159*subarray1*power)/N;\n"
		"tw_real=native_cos(tmp);\n"
		"tw_imag=-1*native_sin(tmp);\n"
		"imag=data_imag_d_in[subarray_start+thread_position];\n"
		"imag2=data_imag_d_in[subarray2_start+thread_position];\n"
		"tmp=real+real2;\n"
		"real2=real-real2;\n"
		"real=tmp;\n"
		"data_real_d_out[subarray_start*2+thread_position]=real;\n"
		"tmp=imag+imag2;\n"
		"imag2=imag-imag2;\n"
		"imag=tmp;\n"
		"tmp=real2;\n"
		"real2=real2*tw_real-imag2*tw_imag;\n"
		"data_real_d_out[subarray_start*2+thread_position+power]=real2;\n"
		"imag2=tmp*tw_imag+imag2*tw_real;\n"
		"data_imag_d_out[subarray_start*2+thread_position]=imag;\n"
		"data_imag_d_out[subarray_start*2+thread_position+power]=imag2;\n"
		"}\n";
//creating program and kernel,also providing compiler flags
	cl_program clpgm;
	clpgm=clCreateProgramWithSource(clctx,1,&kernel,NULL,&clerr);
	char clcompileflags[4096];
	sprintf_s(clcompileflags,"-DUNROLLX=%d -cl-fast-relaxed-math -cl-single-precision-constant -cl-denorms-are-zero -cl-mad-enable",0);
	clerr=clBuildProgram(clpgm,0,NULL,NULL,NULL,NULL);
	cl_kernel clkern=clCreateKernel(clpgm,"stfft",&clerr);
//create device buffers and copy data to them(the flag CL_MEM_COPY_HOST_PTR does that)
	cl_mem real_out=clCreateBuffer(clctx,CL_MEM_READ_WRITE,sizeof(float)*N,NULL,&clerr);
	cl_mem imag_out=clCreateBuffer(clctx,CL_MEM_READ_WRITE,sizeof(float)*N,NULL,&clerr);
	cl_mem real_in=clCreateBuffer(clctx,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,sizeof(float)*N,data_real,&clerr);
	cl_mem imag_in=clCreateBuffer(clctx,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,sizeof(float)*N,data_imag,&clerr);
	cl_mem tmp;
//threads configuration
	const size_t local_ws = 512;
	const size_t global_ws =N/2;
	int length=N;
	long int before = GetTickCount();
//this loop takes care of the multiple launches and buffers swapping
	for(int i=0;i<passes;i++)
	{
		
		clerr=clSetKernelArg(clkern,0,sizeof(cl_mem),&real_in);
		clerr=clSetKernelArg(clkern,1,sizeof(cl_mem),&imag_in);
		clerr=clSetKernelArg(clkern,2,sizeof(cl_mem),&real_out);
		clerr=clSetKernelArg(clkern,3,sizeof(cl_mem),&imag_out);
		clerr=clSetKernelArg(clkern,4,sizeof(int),(void*)&i);
		clerr=clSetKernelArg(clkern,5,sizeof(int),(void*)&length);
		//this line is the kernel launch
		clerr=clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
		tmp=real_in;
		real_in=real_out;
		real_out=tmp;
		tmp=imag_in;
		imag_in=imag_out;
		imag_out=tmp;

	}
long int after = GetTickCount();
printf("time=%ld ms\n",after-before);
	//read back results
	clEnqueueReadBuffer(clcmdq, real_in, CL_FALSE, 0, sizeof(float)*N, data_real, 0, NULL, NULL);
	clEnqueueReadBuffer(clcmdq, imag_in, CL_FALSE, 0, sizeof(float)*N, data_imag, 0, NULL, NULL);

//test results
for(int i=N-16;i<N;i++)
	{
		printf("data[%d]=%f + %f i\n",i,data_real[i],data_imag[i]);
	}

//cleanup
clReleaseKernel(clkern);
clReleaseCommandQueue(clcmdq);
clReleaseContext(clctx);
clReleaseMemObject(real_in);
clReleaseMemObject(real_out);
clReleaseMemObject(imag_in);
clReleaseMemObject(imag_out);

getchar();
return 0;
}