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
#include <openacc.h>
#include <stdio.h>
#include "timer.h"

#define NN 4096
#define NM 4096

double A[NN][NM];
double Anew[NN][NM];

int main(int argc, char** argv)
{
    const int n = NN;
    const int m = NM;
    const int iter_max = 1000;
    
    const double tol = 1.0e-6;
    double error     = 1.0;
    
    //Fill A and Anew with zeors
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));

    // Initialize elements in A and Anew    
    for (int j = 0; j < n; j++)
    {
        A[j][0]    = 1.0;
        Anew[j][0] = 1.0;
    }
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    StartTimer();
    int iter = 0;

// Copy A and Anew from host CPU to GPU (device) memory    
#pragma acc data copyin(Anew), copy(A) 
    while ( error > tol && iter < iter_max ) // Iterate until convergence or reaching the maximum iteration
    {
        error = 0.0;

// Parallelize jacobi iteration with OpenACC Kernel
#pragma acc kernels
        // Iterate across matrix elements
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                // Calculate the new value from neighboring values
                Anew[j][i] = 0.25 * ( A[j][i+1] + A[j][i-1]
                                    + A[j-1][i] + A[j+1][i]);

                // Calculate error as the difference between new and old values to determine convergence 
                error = fmax( error, fabs(Anew[j][i] - A[j][i]));
            }
        }
// Replace A with the computed Anew. Parallelization with OpenACC done on the iteration across matrix elements.         
#pragma acc kernels
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                A[j][i] = Anew[j][i];    
            }
        }

        // Print iteration counts and corresponding errors for every 100 iterations
        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    // Time the whole process
    double runtime = GetTimer();
    printf(" total: %f s\n", runtime / 1000);
}
