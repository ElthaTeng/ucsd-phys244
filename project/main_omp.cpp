# include <cstdlib>
# include <cmath>
# include <fstream>
# include <iostream>
# include <sstream>
# include <string>
# include <iomanip>
# include <omp.h>
# include <chrono>

using namespace std;
using namespace std::chrono;

int main ( int argc, char *argv[] );

// Function that generates the input files for each combination of parameter set
void write_input(int idx_N, int idx_T, int idx_n){
  ofstream myfile;
  int T_value = idx_T * 5 + 10; // Temperatures are set to vary from 10 to 100 K with step = 5
  float N_value = idx_N * 0.1 + 16; // Column densities are set to vary from 16 to 20 cm-2 in log space with step = 0.1 dex
  float n_value = idx_n * 0.1 + 2; // Volume densities are set to vary from 2 to 5 cm-3 in log space with step = 0.1 dex
  
  stringstream ss_N, ss_T, ss_n, ss_pow_N, ss_pow_n;
  ss_N << fixed << setprecision(1) << N_value;
  ss_T << T_value;
  ss_n << fixed << setprecision(1) << n_value;
  ss_pow_N << floor(N_value);
  ss_pow_n << floor(n_value);

  string str_N = ss_N.str();
  string str_T = ss_T.str();
  string str_n = ss_n.str();
  string pow_N = ss_pow_N.str();
  string pow_n = ss_pow_n.str();
  string pre_N = to_string(pow(10, N_value - floor(N_value)));
  string pre_n = to_string(pow(10, n_value - floor(n_value)));
  
  myfile.open ("input/" + str_N + "_" + str_T + "_"+ str_n + ".inp");
  myfile << "co.dat \n"; 
  myfile << "output/" + str_N + "_" + str_T + "_"+ str_n + ".out \n"; 
  myfile << "100 300 \n";
  myfile << str_T + "\n";
  myfile << "1 \n"; 
  myfile << "H2 \n";
  myfile << pre_n + "e" + pow_n + "\n"; 
  myfile << "2.73 \n";
  myfile << pre_N + "e" + pow_N + "\n";
  myfile << "15 \n";
  myfile << "0 \n";
  myfile.close();
}

// Function that computes the predicted observed quantities by executing *RADEX*
void compute(int idx_N, int idx_T, int idx_n){
  int T_value = idx_T * 5 + 10;
  float N_value = idx_N * 0.1 + 16;
  float n_value = idx_n * 0.1 + 2;
  
  stringstream ss_N, ss_T, ss_n;
  ss_N << fixed << setprecision(1) << N_value;
  ss_T << T_value;
  ss_n << fixed << setprecision(1) << n_value;

  string str_N = ss_N.str();
  string str_T = ss_T.str();
  string str_n = ss_n.str();  

  system( ("radex < input/" + str_N + "_" + str_T + "_"+ str_n + ".inp").c_str() );

}

// Function that returns the predicted line flux of the CO 1-0 transition
float read_flux(int idx_N, int idx_T, int idx_n){
  int T_value = idx_T * 5 + 10;
  float N_value = idx_N * 0.1 + 16;
  float n_value = idx_n * 0.1 + 2;
  
  int n_lines = 13; // Number of lines to skip in the output files from step 2
  float flux;
  ifstream file_radex;
  string line;

  stringstream ss_N, ss_T, ss_n;
  ss_N << fixed << setprecision(1) << N_value;
  ss_T << T_value;
  ss_n << fixed << setprecision(1) << n_value;

  string str_N = ss_N.str();
  string str_T = ss_T.str();
  string str_n = ss_n.str();  
  
  file_radex.open("output/" + str_N + "_" + str_T + "_"+ str_n + ".out");
  for (int i = 0 ; i < n_lines ; i++){
    file_radex.ignore(150, '\n'); // Skip the first n_line lines
  } 
  file_radex.ignore(102, '\n'); // Skip the first 102 characters to find the line flux of CO 1-0
  file_radex >> flux;  // Assign the predicted CO 1-0 flux to the variable 'flux' 
  file_radex.close();

  return flux;
}

// Function that computes a probability of good fit with data
float fitting(float flux_mod, float flux_obs, float err){
  float chi2;
  float prob;

  chi2 = pow((flux_mod - flux_obs) / err, 2);
  prob = exp(-0.5 * chi2);
  
  return prob;
}

//****************************************************************************80

int main ( int argc, char *argv[] )
{
  double wtime;
  ofstream outfile;
  
  int size_Nco = 41;
  int size_Tk = 19;
  int size_nH2 = 31;

  float flux_obs = 87.1; // assume a random observed flux data
  float model[size_Nco][size_Tk][size_nH2];
  float probability[size_Nco][size_Tk][size_nH2];

  // Write info into a output log file
  outfile.open ("main_omp.log");
  outfile << "\n";
  outfile << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
  outfile << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";

  wtime = omp_get_wtime ( );

  auto start = high_resolution_clock::now();


  // Step 1: Write input files
  # pragma omp parallel for
  for (int k = 0; k < size_Nco; k++){
    # pragma omp parallel for
    for (int i = 0; i < size_Tk; i++){
      # pragma omp parallel for
      for (int j = 0; j < size_nH2; j++){
        write_input(k,i,j);
      }
    }
  }

  auto stop_1 = high_resolution_clock::now();
  auto duration_1 = duration_cast<microseconds>(stop_1 - start);
  outfile << "Step 1 time: " << duration_1.count() << " microsec \n"; 

  
  // Step 2: Compute theoretical quantities 
  # pragma omp parallel for
  for (int k = 0; k < size_Nco; k++){
    # pragma omp parallel for
    for (int i = 0; i < size_Tk; i++){
      # pragma omp parallel for
      for (int j = 0; j < size_nH2; j++){
        compute(k,i,j);
      }
    }
  }    

  auto stop_2 = high_resolution_clock::now();
  auto duration_2 = duration_cast<microseconds>(stop_2 - stop_1);
  outfile << "Step 2 time: " << duration_2.count() << " microsec \n"; 


  // Step 3: Construct model array by extracting desired outputs from step 2
  # pragma omp parallel for
  for (int k = 0; k < size_Nco; k++){
    # pragma omp parallel for
    for (int i = 0; i < size_Tk; i++){
      # pragma omp parallel for
      for (int j = 0; j < size_nH2; j++){
        model[k][i][j] = read_flux(k,i,j);
      }
    }
  }  

  cout << model[10][15][20] << "\n";
  
  auto stop_3 = high_resolution_clock::now();
  auto duration_3 = duration_cast<microseconds>(stop_3 - stop_2);
  auto duration_tot = duration_cast<microseconds>(stop_3 - start);

  //Step 4: Fitting data with the constructed model
  # pragma omp parallel for
    for (int k = 0; k < size_Nco; k++){
      # pragma omp parallel for
      for (int i = 0; i < size_Tk; i++){
        # pragma omp parallel for
        for (int j = 0; j < size_nH2; j++){
          probability[k][i][j] = fitting(model[k][i][j], flux_obs, 0.1*flux_obs); //assume a 10% flux uncertainty
        }
      }
    }

  cout << probability[10][15][20] << "\n"; 

  auto stop_4 = high_resolution_clock::now();
  auto duration_4 = duration_cast<microseconds>(stop_4 - stop_3);


  outfile << "Step 3 time: " << duration_3.count() << " microsec \n"; 
  outfile << "Total time: " << duration_tot.count() << " microsec \n"; 
  outfile << "Step 4 time: " << duration_4.count() << " microsec \n";

  wtime = omp_get_wtime ( ) - wtime;
  outfile << "Elapsed time = " << wtime << " sec \n";

  outfile.close();
  
  return 0;
}
