/*
 *  Copyright 2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include "timer.h"

#define NN 4096
#define NM 4096

const int n = NN;
const int m = NM;

// Write jacobi into a function that can be called by the host and run on the devicce
__global__ void kernel_jacobi(double *A, double *Anew){
    
    unsigned int rowinit = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int colinit = threadIdx.y + blockDim.y * blockIdx.y;
    int rowstride = gridDim.x * blockDim.x;
    int colstride = gridDim.y * blockDim.y;

    const int iter_max = 1000;
    
    const double tol = 1.0e-6;
    double error     = 1.0;
    int iter = 0;
    
    while ( error > tol && iter < iter_max )
    {
        
        error = 0.0;
        
        for( int row = 1 + rowinit; row < n-1; row += rowstride )
        {
            for( int col = 1 + colinit; col < m-1; col += colstride )
            {
                /*Anew[row][col] = 0.25 * ( A[row][col+1] + A[row][col-1]
                                    + A[row-1][col] + A[row+1][col]);
                error = fmax( error, fabs(Anew[row][col] - A[row][col]));
                */

                Anew[row * m + col] = 0.25 * ( A[row * m + col+1] + A[row * m + col-1]
                                    + A[(row-1) * m + col] + A[(row+1) * m + col]);
                error = fmax( error, fabs(Anew[row * m + col] - A[row * m + col]));

                                
            }
        }

        //Waiting for each thread to finish
        __syncthreads();
        

        for( int row = 1 + rowinit; row < n-1; row += rowstride )
            {
                for( int col = 1 + colinit; col < m-1; col += colstride )
                {
                    //A[row][col] = Anew[row][col];  
                    A[row * m + col] = Anew[row * m + col];  
                }
            }

        //Waiting for each thread to finish
        __syncthreads();
        
        if(iter % 100 == 0) printf("%5d, %0.6f from thread [%d,%d]\n", iter, error, blockIdx.x, threadIdx.x);
        
        iter++;
    }

    return;

}


int main(int argc, char** argv)
{
    double A[NN][NM];
    double *d_A; // device copy

    double Anew[NN][NM];
    double *d_Anew; // device copy

    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));
    
    // Allocate device memory for d_A and d_Anew
    cudaMalloc(&d_A, n * m * sizeof(double));
    cudaMalloc(&d_Anew, n * m * sizeof(double));
        
    for (int j = 0; j < n; j++)
    {
        A[j][0]    = 1.0;
        Anew[j][0] = 1.0;
    }
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    StartTimer();
    
    // Copy A and Anew into device
    cudaMemcpy(d_A, A, n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, Anew, n * m * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the jacobi kernal with 4, 8, 16, 32, and 8192 threads respectively
    kernel_jacobi<<<2,2>>>(d_A, d_Anew);
    //kernel_jacobi<<<2,4>>>(d_A, d_Anew);
    //kernel_jacobi<<<4,4>>>(d_A, d_Anew);
    //kernel_jacobi<<<4,8>>>(d_A, d_Anew);
    //kernel_jacobi<<<64,128>>>(d_A, d_Anew);

    // Copy results back to host (just A, no need to copy Anew back)
    cudaMemcpy(A, d_A, n * m * sizeof(double), cudaMemcpyDeviceToHost);

    // Deallocate memory to free up device space
    cudaFree(d_A);
    cudaFree(d_Anew);

    double runtime = GetTimer();
 
    printf(" total: %f s\n", runtime / 1000);
}
