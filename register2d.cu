#include <cuda.h>
#include <math.h>

__global__ void extf( double *out, double *img1, double *img2, double *grad, int rows, int columns )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// if (i >= columns || j >= rows)
	// 	return;
	
	out[i + j * columns] = 2 * (img1[i + j * columns] - img2[i + j * columns]) * grad[i + j * columns];
	
}

__global__ void intf(
	double *out,
	double *f,
	int rows,
	int columns)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < 1 || j < 1 || i  >= (columns - 1) || j >= (rows - 1))
		return;

	out[i * rows + j] = -4 * f[i * rows + j] + f[(i - 1) * rows + j] + f[(i + 1) * rows +j] + f[i * rows + (j - 1)] + f[i * rows + (j+ 1)];
}

__global__ void add(
	double *out,
	double *in1,
	double *in2,
	int rows,
	int columns)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < 2 || j < 2 || i  >= (columns - 2) || j >= (rows - 2))
		return;

	out[i * rows + j] = in1[i * rows + j] + in2[i * rows + j];
}


__global__ void d_f(
	double *out,
	double *intf,
	double *extf,
	double rho,
	double lambda,
	int rows,
	int columns)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < 1 || j < 1 || i  >= (columns - 1) || j >= (rows - 1))
		return;

	out[i * rows + j] = rho * (extf[i * rows + j] + lambda * intf[i * rows + j]);
}