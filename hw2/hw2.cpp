# include <cstdlib>
# include <cmath>
# include <fstream>
# include <iostream>
# include <iomanip>
# include <omp.h>

using namespace std;

int main ( int argc, char *argv[] );

//****************************************************************************80

int main ( int argc, char *argv[] )
{
  double wtime;
  ofstream myfile;
  
  int size = 10001;
  double k = 0.00062831853;
  double alpha = 0.9;
  double y0[size];
  double yt0[size];
  double y1[size];
  double yt1[size];

cout << "\n";
cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";

wtime = omp_get_wtime ( );

// Initialization
  # pragma omp parallel
  {
    # pragma omp for
    for (int i = 0; i < size; i++)
    {
      yt0[i] = 0;

      if ( (i == 0) || (i==10000) ) // first and last elements are always zeros
      {
        y0[i] = 0;
        y1[i] = 0;
        yt1[i] = 0;  
      }
      else
      {
        y0[i] = sin (k * i);      
      } 
    }
  }

// Iterate for 1000 steps to get y1000 and yt1000
  for (int step = 0; step < 1000; step++)
  {
    // Compute y1 and yt1 based on y0 and yt0
    # pragma omp parallel
    {
      # pragma omp for
      for (int i = 1; i < size-1; i++)
      {
        y1[i] = 0.5 * pow(alpha, 2) * y0[i-1] + (1.0 - pow(alpha, 2)) * y0[i] + 0.5 * pow(alpha, 2) * y0[i+1] + yt0[i];
        yt1[i] = y1[i] - y0[i];
      }
    }

    // Set y0 = y1 and yt0 = yt1 to compute the vectors for next step
    # pragma omp parallel
    {
      # pragma omp for
      for (int i = 0; i < size; i++)
      {
        y0[i] = y1[i];
        yt0[i] = yt1[i];
      }
    }
  }

wtime = omp_get_wtime ( ) - wtime;
cout << "Elapsed time = " << wtime << "\n";

// Write the final y1000 vector to a txt file 
  myfile.open ("y1000.txt");
  myfile << "y1000 = \n"; 
  for (int i = 0; i < size; i++)
  {
    myfile << y1[i] << "\n";
  }
  myfile.close();

  return 0;
}
