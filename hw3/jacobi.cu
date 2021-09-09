#include <iostream>
#include <fstream>

#define N_ROWS 27
#define N_COLUMNS 27
#define PRECISION 1.e-4 // ERROR

__device__ bool lIsFinished = false;

__host__ __device__ void print_matrix(const float *aMatrix) {
  printf(" --- MATRIX --- \n");
  for (unsigned int lRow = 0; lRow < N_ROWS; lRow++)
  {
    for (unsigned int lColumn = 0; lColumn < N_COLUMNS; lColumn++)
    {
      printf("%.2f",  aMatrix[lRow * N_COLUMNS + lColumn]);
      if(lColumn < N_COLUMNS - 1) printf(",");
    }
    printf("\n");
  }
}

__global__ void kernel_jacobi(float* aMatrix1, float* aMatrix2, unsigned int aNumberOfRows, unsigned int aNumberOfColumns, float aPrecision)
{
  unsigned int lThreadIndexX = blockIdx.x * gridDim.x + threadIdx.x;
  unsigned int lThreadIndexY = blockIdx.y * gridDim.y + threadIdx.y;
  unsigned int lNumberOfNeighbours = 0;
  float lNewFieldValue = 0;

  while(!lIsFinished) {
    //Passing from matrix 1 to matrix 2
    lIsFinished = true;
    
    for(lThreadIndexY = blockIdx.y * gridDim.y + threadIdx.y; lThreadIndexY < aNumberOfRows; lThreadIndexY += blockDim.y * gridDim.y)
    {
      for(lThreadIndexX = blockIdx.x * gridDim.x + threadIdx.x; lThreadIndexX < aNumberOfColumns; lThreadIndexX += blockDim.x * gridDim.x)
      {
        //if ((lThreadIndexY == N_ROWS/2) && (lThreadIndexX == N_COLUMNS/2)) continue;
        if (lThreadIndexY == 0) continue;
        if (lThreadIndexY == N_ROWS - 1) continue;
        lNewFieldValue = 0;
        lNumberOfNeighbours = 0;
        //If we are not in the top row, an upper neighbour exists
        if (lThreadIndexY > 0) {
          lNewFieldValue += aMatrix1[(lThreadIndexY - 1) * aNumberOfColumns + lThreadIndexX];
          lNumberOfNeighbours++;
        }
        //Check if we are in the bottom row
        if (lThreadIndexY < (aNumberOfRows - 1) ) {
          lNewFieldValue += aMatrix1[(lThreadIndexY + 1) * aNumberOfColumns + lThreadIndexX];
          lNumberOfNeighbours++;
        }
        //leftmost column
        if (lThreadIndexX > 0) {
          lNewFieldValue += aMatrix1[lThreadIndexY  * aNumberOfColumns + lThreadIndexX - 1];
          lNumberOfNeighbours++;
        }
        //rightmost column
        if (lThreadIndexX < (aNumberOfColumns - 1) ) {
          lNewFieldValue += aMatrix1[lThreadIndexY * aNumberOfColumns + lThreadIndexX + 1];
          lNumberOfNeighbours++;
        }
        //Calculating the average
        lNewFieldValue /= lNumberOfNeighbours;
        //Assigning the found value to the new matrix
        aMatrix2[lThreadIndexY * aNumberOfColumns + lThreadIndexX] = lNewFieldValue;
      }
    }
    //Waiting for each thread to finish
    __syncthreads();

    //print_matrix(aMatrix2);
    
    //Passing from matrix 2 to matrix 1
    for(lThreadIndexY = blockIdx.y * gridDim.y + threadIdx.y; lThreadIndexY < aNumberOfRows; lThreadIndexY += blockDim.y * gridDim.y)
    {
      for(lThreadIndexX = blockIdx.x * gridDim.x + threadIdx.x; lThreadIndexX < aNumberOfColumns; lThreadIndexX += blockDim.x * gridDim.x)
      {
        //if ((lThreadIndexY == N_ROWS/2) && (lThreadIndexX == N_COLUMNS/2)) continue;
        if (lThreadIndexY == 0) continue;
        if (lThreadIndexY == N_ROWS - 1) continue;
        lNewFieldValue = 0;
        lNumberOfNeighbours = 0;
        //If we are not in the top row, an upper neighbour exists
        if (lThreadIndexY > 0) {
          lNewFieldValue += aMatrix2[(lThreadIndexY - 1) * aNumberOfColumns + lThreadIndexX];
          lNumberOfNeighbours++;
        }
        //Check if we are in the bottom row
        if (lThreadIndexY < (aNumberOfRows - 1) ) {
          lNewFieldValue += aMatrix2[(lThreadIndexY + 1) * aNumberOfColumns + lThreadIndexX];
          lNumberOfNeighbours++;
        }
        //leftmost column
        if (lThreadIndexX > 0) {
          lNewFieldValue += aMatrix2[lThreadIndexY * aNumberOfColumns + lThreadIndexX - 1];
          lNumberOfNeighbours++;
        }
        //rightmost column
        if (lThreadIndexX < (aNumberOfColumns - 1) ) {
          lNewFieldValue += aMatrix2[lThreadIndexY * aNumberOfColumns + lThreadIndexX + 1];
          lNumberOfNeighbours++;
        }
        //Calculating the average
        lNewFieldValue /= lNumberOfNeighbours;
        //Assigning the found value to the new matrix
        aMatrix1[lThreadIndexY * aNumberOfColumns + lThreadIndexX] = lNewFieldValue;
        //Checking if we are precise enough
        if (fabsf(lNewFieldValue - aMatrix2[lThreadIndexY * aNumberOfColumns + lThreadIndexX]) > PRECISION) lIsFinished = false;
      }
    }
    //Waiting for each thread to finish
    __syncthreads();
    //print_matrix(aMatrix1);
  }

  return;
}


int main(int aArgc, char* aArgv[])
{
  //Initialisation of the field
  float lMatrix[N_ROWS][N_COLUMNS];

  //Setting every element to 0
  for (unsigned int lRow = 0; lRow < N_ROWS; lRow++)
  {
    for (unsigned int lColumn = 0; lColumn < N_COLUMNS; lColumn++)
    {
      lMatrix[lRow][lColumn] = 0;
      if (lRow == 0)lMatrix[lRow][lColumn] = 10;
      if (lRow == N_ROWS - 1)lMatrix[lRow][lColumn] = 0;
    }
  }

  //Creating a central charge
  //lMatrix[N_ROWS/2][N_COLUMNS/2] = 10.f;

  // --- Preparing the GPU ---
  //Allocating the first matrix
  float* lD_Matrix1 = NULL;
  unsigned int lMemSize = N_ROWS * N_COLUMNS * sizeof(float);
  cudaMalloc(&lD_Matrix1, lMemSize);
  cudaMemcpy(lD_Matrix1, lMatrix, lMemSize, cudaMemcpyHostToDevice);
  //Allocating the second matrix
  float* lD_Matrix2 = NULL;
  cudaMalloc(&lD_Matrix2, lMemSize);
  cudaMemcpy(lD_Matrix2, lMatrix, lMemSize, cudaMemcpyHostToDevice);

  //Calling the CUDA kernel

  //dim3 lGridSize(32, 32, 1);
  //dim3 lBlockSize(32, 32, 1);
  dim3 lGridSize(1, 1, 1);
  dim3 lBlockSize(1, 1, 1);
  kernel_jacobi<<<lGridSize, lBlockSize>>>(lD_Matrix1, lD_Matrix2, N_ROWS, N_COLUMNS, PRECISION);

  //Copying the result
  cudaMemcpy(lMatrix, lD_Matrix1, lMemSize, cudaMemcpyDeviceToHost);

  //Printing it
  print_matrix((const float *) lMatrix);

  cudaFree(lD_Matrix1);
  cudaFree(lD_Matrix2);
  
  return 0;
}
