# Barnes-Hut 2 GPU Simulation
  
This program is an expansion upon the work done by Martin Burtscher and Keshav Pingali. Their original implementation simulates an N-Body program making use of a GPU to accelerate the calculations. In this program, the code is expanded to make use of a second GPU in order to accelerate the simulations even more. Their published paper can be found at `http://iss.ices.utexas.edu/Publications/Papers/burtscher11.pdf` and the original source code can be found at `http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu`.

### Prerequisites

This program requires CUDA to be installed on the machine, as well as two CUDA-enabled GPU's with compute capability 2.0 or above

To install CUDA, Nvidia has provided installation guides for the main operating systems:

-Windows:  https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html <br />
-Mac OS X: https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html <br />
-Linux:    https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html <br />

Nvidia has also provided a list of CUDA-enabled GPU's with their respective compute capabilities:

GPU list: `<https://developer.nvidia.com/cuda-gpus>`

### Compiling the codes

A Makefile is available. If all prerequisites are met, simply go to the directory in which the files are located and type:

```make```

## Running the program

Execute the _bh_ program by typing:

```$./bh <Number Of Bodies> <Number Of Time Steps> <Main GPU Nr> <Auxiliary GPU Nr>```

Note: To find the number associated with each GPU, run Nvidia's _deviceQuery_ program. More details can be found in the installation guides above.

Example:

```$./bh 10000 100 0 1```