# PHYS 244 Final Project - Implementing Astrophysical Modeling Pipelines with CPU and GPU Parallelization

## Prequisites

* Install RADEX simply by unzip/untar the distribution file
    * Distribution file: https://personal.sron.nl/~vdtak/radex/radex_public.tar.gz
    * Instructions: https://personal.sron.nl/~vdtak/radex/index.shtml
* Place the molecular data file *co.dat* under *Radex/data*
* Create RADEX input and output directories for both models:
    * *mkdir input*
    * *mkdir input_coarse*
    * *mkdir output*
    * *mkdir output_coarse*

## Code Compilation / Execution

* Load required modules: *slurm*, *cpu* or *gpu*, *gcc* or *pgi*
    * e.g. *module load slurm*

* Compile source codes: *main.cpp*, *main_omp.cpp*, *main_acc.cpp*, *main_coarse.cpp*, *main_omp_coarse.cpp*, *main_acc_coarse.cpp* 
    * Serial codes: *g++ -o main main.cpp* 
    * OpenMP codes: *g++ -fopenmp -o main_omp main_omp.cpp*  
    * OpenACC codes: *g++ -fopenacc -o main_acc main_acc.cpp*  

* For serial codes, execute directly on command line
    * *./main* or *./main_coarse*

* For parallelized codes, submit batch scripts to Slurm: 
    * OpenMP codes: *sbatch omp_threadx.sb* where *x* = number of threads
    * OpenACC codes: *sbatch openacc.sb*


## Outputs

  * RADEX input files
    * saved to the prepared input directories as e.g. *16.6_80_4.6.inp*; values are in the order of log(N<sub>CO</sub>), T<sub>k</sub>, and log(n<sub>H<sub>2</sub></sub>)
  * RADEX output files
    * saved to the prepared output directories as e.g. *17.0_25_4.2.out*; same naming format as the RADEX input files
  * output log files
    * saved to the current directory as e.g. *main.log*; contains information on execution time for each step and number of threads/processors available 