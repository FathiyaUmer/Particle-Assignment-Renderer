/*
    Author: Fathiya
*/
#include "cuda.cuh"
#include "helper.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstring>
#include <cmath>

// Algorithm storage
// Number of particles in d_particles
unsigned int cuda_particles_count;

// Device pointer to a list of particles
Particle* d_particles;

// VALIDATION CPU pointer to a list of particles for validation
Particle* cpu_particles;

// Device pointer to a histogram of the number of particles contributing to each pixel
unsigned int* d_pixel_contribs;

// VALIDATION CPU pointer to a histogram of the number of particles contributing to each pixel 
unsigned int* cpu_pixel_contribs;

// Device pointer to an index of unique offsets for each pixels contributing colours
unsigned int* d_pixel_index;

// VALIDATION pointer to an index of unique offsets for each pixels contributing colours 
unsigned int* cpu_pixel_index;

// Device pointer to storage for each pixels contributing colours
unsigned char* d_pixel_contrib_colours;

// Host pointer to storage for each pixels contributing colours
unsigned char* cpu_pixel_contrib_colours;

// Device pointer to storage for each pixels contributing colours' depth
float* d_pixel_contrib_depth;

// Host pointer to storage for each pixels contributing colours' depth
float* cpu_pixel_contrib_depth;

// The number of contributors d_pixel_contrib_colours and d_pixel_contrib_depth have been allocated for
unsigned int cuda_pixel_contrib_count;

// Host storage of the output image dimensions
int cuda_output_image_width;
int cuda_output_image_height;

// Device storage of the output image dimensions
__constant__ int D_OUTPUT_IMAGE_WIDTH;
__constant__ int D_OUTPUT_IMAGE_HEIGHT;

// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* cpu_output_image_data;

unsigned int TOTAL_CONTRIBS;

CImage cpu_output_image;

void cpu_sort_pairs(float* keys_start, unsigned char* colours_start, int first, int last);

void cuda_begin(const Particle* init_particles, const unsigned int init_particles_count, const unsigned int out_image_width, const unsigned int out_image_height) {
    // These are basic CUDA memory allocations that match the CPU implementation
    // Depending on your optimisation, you may wish to rewrite these (and update cuda_end())

    // Allocate a opy of the initial particles, to be used during computation
    cuda_particles_count = init_particles_count;
    CUDA_CALL(cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles, init_particles_count * sizeof(Particle), cudaMemcpyHostToDevice));

    // Allocate a histogram to track how many particles contribute to each pixel
    CUDA_CALL(cudaMalloc(&d_pixel_contribs, out_image_width * out_image_height * sizeof(unsigned int)));

    // Allocate an index to track where data for each pixel's contributing colour starts/ends
    CUDA_CALL(cudaMalloc(&d_pixel_index, (out_image_width * out_image_height + 1) * sizeof(unsigned int)));

    // Init a buffer to store colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_colours = 0;

    // Init a buffer to store depth of colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_depth = 0;

    // This tracks the number of contributes the two above buffers are allocated for, init 0
    cuda_pixel_contrib_count = 0;

    // Allocate output image
    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)));
    const int CHANNELS = 3;  // RGB
    CUDA_CALL(cudaMalloc(&cpu_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char)));

    // Allocate memory for variables used in validation
    cpu_particles = (Particle*)malloc(sizeof(Particle) * init_particles_count);

    memcpy(cpu_particles, init_particles, (sizeof(Particle) * init_particles_count));

    cpu_pixel_contribs = (unsigned int*)malloc(sizeof(unsigned int) * (out_image_width * out_image_height));
    cpu_pixel_index = (unsigned int*)malloc(sizeof(unsigned int) * (out_image_width * out_image_height + 1));
    cpu_output_image.channels = 3;
    TOTAL_CONTRIBS = 0;
    cpu_output_image.data = (unsigned char*)malloc(cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char));
}


// Stage 1 kernel implementation
__global__ void stage1_calculate_histogram_kernel(const Particle* particles, int image_width, int image_height, unsigned int* device_histogram)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate co-ordinates of bounding box around the particle
    int x_min = (int)roundf(particles[threadIndex].location[0] - particles[threadIndex].radius);
    int y_min = (int)roundf(particles[threadIndex].location[1] - particles[threadIndex].radius);
    int x_max = (int)roundf(particles[threadIndex].location[0] + particles[threadIndex].radius);
    int y_max = (int)roundf(particles[threadIndex].location[1] + particles[threadIndex].radius);

    // Clamp bounding box to image bounds
    x_min = x_min < 0 ? 0 : x_min;
    y_min = y_min < 0 ? 0 : y_min;
    x_max = x_max >= image_width ? image_width - 1 : x_max;
    y_max = y_max >= image_height ? image_height - 1 : y_max;

    // For each pixel in the bounding box, check that it falls within the radius
    for (int x = x_min; x <= x_max; ++x)
    {
        for (int y = y_min; y <= y_max; ++y)
        {
            const float x_ab = (float)x + 0.5f - particles[threadIndex].location[0];
            const float y_ab = (float)y + 0.5f - particles[threadIndex].location[1];
            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
            if (pixel_distance <= particles[threadIndex].radius)
            {
                const unsigned int pixel_offset = y * image_width + x;
                // Handling Race Condition using atomicAdd() cuda function.
                atomicAdd(&device_histogram[pixel_offset], 1);
            }
        }
    }
}


void cuda_stage1() {
    stage1_calculate_histogram_kernel <<<(cuda_particles_count / 1024) + 1, 1024 >> > (d_particles, cuda_output_image_width, cuda_output_image_height, d_pixel_contribs);
    cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(cpu_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(int), cudaMemcpyDeviceToHost));
    
#ifdef VALIDATION
    // Validation of histogram
    validate_pixel_contribs(cpu_particles, cuda_particles_count, cpu_pixel_contribs, cuda_output_image_width, cuda_output_image_height);
#endif
}


__global__ void stage2_store_color_depth(Particle* d_particles, unsigned int* d_pixel_contribs, unsigned int* d_pixel_index, int image_width, int image_height, unsigned char* d_pixel_contrib_colours, float* d_pixel_contrib_depth, unsigned int particle_count)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIndex < particle_count)
    {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(d_particles[threadIndex].location[0] - d_particles[threadIndex].radius);
        int y_min = (int)roundf(d_particles[threadIndex].location[1] - d_particles[threadIndex].radius);
        int x_max = (int)roundf(d_particles[threadIndex].location[0] + d_particles[threadIndex].radius);
        int y_max = (int)roundf(d_particles[threadIndex].location[1] + d_particles[threadIndex].radius);

        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= image_width ? image_width - 1 : x_max;
        y_max = y_max >= image_height ? image_height - 1 : y_max;

        // Store data for every pixel within the bounding box that falls within the radius
        for (int x = x_min; x <= x_max; ++x)
        {
            for (int y = y_min; y <= y_max; ++y)
            {
                const float x_ab = (float)x + 0.5f - d_particles[threadIndex].location[0];
                const float y_ab = (float)y + 0.5f - d_particles[threadIndex].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);

                if (pixel_distance <= d_particles[threadIndex].radius)
                {
                    const unsigned int pixel_offset = y * image_width + x;
                    unsigned int storage_offset = d_pixel_index[pixel_offset];
                    storage_offset += atomicAdd(&d_pixel_contribs[pixel_offset], 1);
                    memcpy(d_pixel_contrib_colours + (4 * storage_offset), d_particles[threadIndex].color, 4 * sizeof(unsigned char));
                    memcpy(d_pixel_contrib_depth + storage_offset, &d_particles[threadIndex].location[2], sizeof(float));
                }
            }
        }
    }
}


// Pair sorting
void cpu_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last) {
    int i, j, pivot;
    float depth_t;
    unsigned char color_t[4];
    if (first < last) {
        pivot = first;
        i = first;
        j = last;
        while (i < j) {
            while (keys_start[i] <= keys_start[pivot] && i < last)
                i++;
            while (keys_start[j] > keys_start[pivot])
                j--;
            if (i < j) {
                // Swap key
                depth_t = keys_start[i];
                keys_start[i] = keys_start[j];
                keys_start[j] = depth_t;
                // Swap color
                memcpy(color_t, colours_start + (4 * i), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * i), colours_start + (4 * j), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
            }
        }
        // Swap key
        depth_t = keys_start[pivot];
        keys_start[pivot] = keys_start[j];
        keys_start[j] = depth_t;
        // Swap color
        memcpy(color_t, colours_start + (4 * pivot), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * pivot), colours_start + (4 * j), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
        // Recurse
        cpu_sort_pairs(keys_start, colours_start, first, j - 1);
        cpu_sort_pairs(keys_start, colours_start, j + 1, last);
    }
}


void cuda_stage2() {

    // Calculate prefix sum array
    cpu_pixel_index[0] = 0;
    for (int i = 0; i < cuda_output_image_width * cuda_output_image_height; ++i) {
        cpu_pixel_index[i + 1] = cpu_pixel_index[i] + cpu_pixel_contribs[i];
    }
    CUDA_CALL(cudaMemcpy(d_pixel_index, cpu_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Allocate memory for colors and depth storage
    TOTAL_CONTRIBS = cpu_pixel_index[cuda_output_image_width * cuda_output_image_height];
    CUDA_CALL(cudaMalloc(&d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char)));
    CUDA_CALL(cudaMalloc(&d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float)));
    cpu_pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
    cpu_pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));

    cudaMemset(d_pixel_contribs, 0, sizeof(int) * (cuda_output_image_width * cuda_output_image_height));
    stage2_store_color_depth << <(cuda_particles_count / 1024) + 1, 1024 >> > (d_particles, d_pixel_contribs, d_pixel_index, cuda_output_image_width, cuda_output_image_height, d_pixel_contrib_colours, d_pixel_contrib_depth, cuda_particles_count);

    CUDA_CALL(cudaMemcpy(cpu_pixel_contrib_colours, d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cpu_pixel_contrib_depth, d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float), cudaMemcpyDeviceToHost));

    // Pair sort the color information for each pixel using quick sort pair sorting version
    for (int i = 0; i < cuda_output_image_width * cuda_output_image_height; ++i) {
        // Pair sort the colours which contribute to a single pigment
        cpu_sort_pairs(
            cpu_pixel_contrib_depth,
            cpu_pixel_contrib_colours,
            cpu_pixel_index[i],
            cpu_pixel_index[i + 1] - 1
        );
    }

#ifdef VALIDATION
    validate_pixel_index(cpu_pixel_contribs, cpu_pixel_index, cuda_output_image_width, cuda_output_image_height);
    validate_sorted_pairs(cpu_particles, cuda_particles_count, cpu_pixel_index, cuda_output_image_width, cuda_output_image_height, cpu_pixel_contrib_colours, cpu_pixel_contrib_depth);
#endif    
}

__global__ void stage3_blending_kernel(unsigned char* device_output_image_data, const unsigned char* __restrict__ d_pixel_contrib_colours, const unsigned int* __restrict__ d_pixel_index)
{
    unsigned int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

    unsigned int start_pixel_index = d_pixel_index[threadIndex];
    unsigned int end_pixel_index = d_pixel_index[threadIndex + 1];

    if (threadIndex < D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT) {
        for (unsigned int i = start_pixel_index; i < end_pixel_index; i++)
        {
            unsigned char cls[4];
            memcpy(cls, d_pixel_contrib_colours + i * 4, 4);
            const float opacity = (float)cls[3] / (float)255;

            device_output_image_data[threadIndex * 3 + 0] = (unsigned char)((float)cls[0] * opacity +
                (float)device_output_image_data[threadIndex * 3 + 0] * (1 - opacity));

            device_output_image_data[threadIndex * 3 + 1] = (unsigned char)((float)cls[1] * opacity +
                (float)device_output_image_data[threadIndex * 3 + 1] * (1 - opacity));

            device_output_image_data[threadIndex * 3 + 2] = (unsigned char)((float)cls[2] * opacity +
                (float)device_output_image_data[threadIndex * 3 + 2] * (1 - opacity));
        }
    }
}


void cuda_stage3()
{
    CUDA_CALL(cudaMemcpy(d_pixel_contrib_colours, cpu_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pixel_index, cpu_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemset(cpu_output_image_data, 255, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char)))
        stage3_blending_kernel << <cuda_output_image_width * cuda_output_image_height / 1024 + 1, 1024 >> > (cpu_output_image_data, d_pixel_contrib_colours, d_pixel_index);

#ifdef VALIDATION
    cpu_pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(cpu_pixel_contrib_colours, d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost))
        CImage v_image;
    v_image.height = cuda_output_image_height;
    v_image.width = cuda_output_image_width;
    v_image.channels = 3;
    v_image.data = (unsigned char*)malloc(
        cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(v_image.data, cpu_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost))
        validate_blend(cpu_pixel_index, cpu_pixel_contrib_colours, &v_image);
#endif
}

void cuda_end(CImage* output_image)
{
    // Store return value
    const int CHANNELS = 3;
    output_image->width = cuda_output_image_width;
    output_image->height = cuda_output_image_height;
    output_image->channels = CHANNELS;

    CUDA_CALL(cudaMemcpy(output_image->data, cpu_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Release allocations
    CUDA_CALL(cudaFree(d_pixel_contrib_depth));
    CUDA_CALL(cudaFree(d_pixel_contrib_colours));
    CUDA_CALL(cudaFree(cpu_output_image_data));
    CUDA_CALL(cudaFree(d_pixel_index));
    CUDA_CALL(cudaFree(d_pixel_contribs));
    CUDA_CALL(cudaFree(d_particles));
    // Return ptrs to nullptr
    d_pixel_contrib_depth = 0;
    d_pixel_contrib_colours = 0;
    cpu_output_image_data = 0;
    d_pixel_index = 0;
    d_pixel_contribs = 0;
    d_particles = 0;

    free(cpu_particles);
    free(cpu_pixel_contribs);
    free(cpu_pixel_index);
    free(cpu_pixel_contrib_colours);
    free(cpu_pixel_contrib_depth);
}

