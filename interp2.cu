#include <cuda.h>
#include <stdio.h>
#include <math.h>

__global__ void interp2(double *imgout, double *fCol, double *fRow, double *imgin, int rows, int cols)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= rows || j >= cols)
		return;

	double i_o_f = fCol[i * cols + j];
	double j_o_f = fRow[i * cols + j];

	i_o_f = fmax(1.0, fmin(i_o_f, (double) cols));
	j_o_f = fmax(1.0, fmin(j_o_f, (double) rows));

	//we will interpolate x direction first, giving R1 and R2//
	double R1 = (floor(i_o_f + 1) - i_o_f) * imgin[(int) floor(i_o_f - 1) * cols + (int) floor(j_o_f - 1)] + (i_o_f - floor(i_o_f)) * imgin[(int) ceil(i_o_f - 1) * cols + (int) floor(j_o_f - 1)];
	double R2 = (floor(i_o_f + 1) - i_o_f) * imgin[(int) floor(i_o_f - 1) * cols + (int) ceil(j_o_f - 1)] + (i_o_f - floor(i_o_f)) * imgin[(int) ceil(i_o_f - 1) * cols + (int) ceil(j_o_f - 1)];

	//now finish//
	imgout[i * cols + j] = (floor(j_o_f + 1) - j_o_f) * R1 + (j_o_f - floor(j_o_f)) * R2;
}