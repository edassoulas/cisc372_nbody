#include <cuda_runtime.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//kernel 1 - will run on GPU executed by many threads at the same time where 
//each thread handles one cell (i,j) of the matrix
// Constructs a NUMENTITIES x NUMENTITIES matrix to hold the pairwise
// acceleration effects between any 2 objects


// used a 1d array laid out in row major order instead of 2d array.
__global__ void computeAccels(vector3* pos, double* mass, vector3* accels, int n) {
    
    //have to compute global indexes for thread
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    //used Claude Sonnet 4.6 to confirm that a 1d flattened array
    //is more efficient than a 2d pointer array for GPU global memory access
    if (i >= n || j >= n) return;

    if (i == j) {
        FILL_VECTOR(accels[i * n + j], 0, 0, 0);
    } else {
	//compute vector pointing from obj j to obj i
        vector3 distance;

        for (int k = 0; k < 3; k++)
            distance[k] = pos[i][k] - pos[j][k];

        double magnitude_sq = distance[0]*distance[0]
                            + distance[1]*distance[1]
                            + distance[2]*distance[2];
        //used claude sonnet 4.6 to verify that magnitude_sq < 1e-12 is the 
        //appropraite epsilon for double precision gravity calculations

        //if too objects are really close together, mangitude_sq approaches 0
	//which would cause div by 0, and with random asteroids its possible

        if (magnitude_sq < 1e-12) {
            FILL_VECTOR(accels[i * n + j], 0, 0, 0);
            return;
        }

        double magnitude = sqrt(magnitude_sq);
        double accelmag = -GRAV_CONSTANT * mass[j] / magnitude_sq;

        FILL_VECTOR(accels[i * n + j],
            accelmag * distance[0] / magnitude,
            accelmag * distance[1] / magnitude,
            accelmag * distance[2] / magnitude);
    }
}

//kernel 2 which runs after kernel 1 is finished and each thread handles one onject (index i)
// adds all accelerations in row i of the accels matrix and uses that to update the velocity of the obnject which uses that velocity to update the position

__global__ void sumAndUpdate(vector3* pos, vector3* vel, vector3* accels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //to ignore the threads beyond num of objects
    if (i >= n) return;

    vector3 accel_sum = {0, 0, 0}; //start with zero acceleration

    //sums up every cell in row i of acceleration matrix
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < 3; k++)
            accel_sum[k] += accels[i * n + j][k];
    }
   
    //interval is the time step whihc is defined in config.h
    for (int k = 0; k < 3; k++) {
        vel[i][k] += accel_sum[k] * INTERVAL;
        pos[i][k] += vel[i][k] * INTERVAL;
    }
}

// static pointers that live on GPU so they persist between calls to compute()
//start as null then get allocated when compute() is called
static vector3* d_pos    = NULL;
static vector3* d_vel    = NULL;
static double*  d_mass   = NULL;
static vector3* d_accels = NULL;
static int allocated_n   = 0;

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL

//added extern "c" with help of claude sonnet 4.6 because my code was breaking with nbody.c
//since nvcc compiles as c++ 
extern "C" void cleanup() {
    if (d_pos) {
        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_mass);
        cudaFree(d_accels);
        d_pos = NULL;
        d_vel = NULL;
        d_mass = NULL;
        d_accels = NULL;
        allocated_n = 0;
    }
}

extern "C" void compute() {
    //total number of planets/asteroids/sun
    int n = NUMENTITIES;

    // gpu memory is allocated once and then reused, if memory was allocated 
    //for a different n, it is freed first
    if (allocated_n != n) {
        if (d_pos) {
            cudaFree(d_pos);
            cudaFree(d_vel);
            cudaFree(d_mass);
            cudaFree(d_accels);
        }

	// allocates memory on gpu
        cudaMalloc(&d_pos,    sizeof(vector3) * n);
        cudaMalloc(&d_vel,    sizeof(vector3) * n);
        cudaMalloc(&d_mass,   sizeof(double)  * n);
        cudaMalloc(&d_accels, sizeof(vector3) * n * n);

        allocated_n = n;
    }

    // Copy host to device (from cpu memory to gpu memory)
    cudaMemcpy(d_pos,  hPos, sizeof(vector3) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel,  hVel, sizeof(vector3) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double)  * n, cudaMemcpyHostToDevice);

    // run kernel 1
    dim3 blockSize(16, 16);
    dim3 gridSize((n + 15) / 16, (n + 15) / 16);
    computeAccels<<<gridSize, blockSize>>>(d_pos, d_mass, d_accels, n);
    cudaDeviceSynchronize();

    //run kernel 
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    sumAndUpdate<<<blocks, threads>>>(d_pos, d_vel, d_accels, n);
    
    //wait for kernel 2 to finish before results are copied to cpu
    cudaDeviceSynchronize();

    // Copy device to host (gpu to cpu)
    // copy updated positions/velocities to cpu
    cudaMemcpy(hPos, d_pos, sizeof(vector3) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_vel, sizeof(vector3) * n, cudaMemcpyDeviceToHost);
}
