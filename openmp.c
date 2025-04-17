#include "openmp.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

void omp_sort_pairs(float* keys_start, unsigned char* colours_start, int first, int last);

/// Algorithm storage
///
unsigned int omp_particles_count;
Particle* omp_particles;
unsigned int* omp_pixel_particle_count_histogram;
unsigned int* omp_pixel_index;
unsigned char* omp_pixel_contrib_colours;
float* omp_pixel_contrib_depth;
unsigned int omp_pixel_contrib_count;
CImage omp_output_image;


void openmp_begin(const Particle* init_particles, const unsigned int init_particles_count, const unsigned int out_image_width, const unsigned int out_image_height)
{
    // Allocate a opy of the initial particles, to be used during computation
    omp_particles_count = init_particles_count;
    omp_particles = malloc(init_particles_count * sizeof(Particle));
    memcpy(omp_particles, init_particles, init_particles_count * sizeof(Particle));

    // Allocate a histogram to track how many particles contribute to each pixel
    omp_pixel_particle_count_histogram = (unsigned int*)malloc(out_image_width * out_image_height * sizeof(unsigned int));

    // Allocate an index to track where data for each pixel's contributing colour starts/ends
    omp_pixel_index = (unsigned int*)malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));

    // Init a buffer to store colours contributing to each pixel into (allocated in stage 2)
    omp_pixel_contrib_colours = 0;

    // Init a buffer to store depth of colours contributing to each pixel into (allocated in stage 2)
    omp_pixel_contrib_depth = 0;

    // This tracks the number of contributes the two above buffers are allocated for, init 0
    omp_pixel_contrib_count = 0;

    // Allocate output image
    omp_output_image.width = (int)out_image_width;
    omp_output_image.height = (int)out_image_height;
    omp_output_image.channels = 3;  // RGB
    omp_output_image.data = (unsigned char*)malloc(omp_output_image.width * omp_output_image.height * omp_output_image.channels * sizeof(unsigned char));
}
void openmp_stage1() {
    memset(omp_pixel_particle_count_histogram, 0, omp_output_image.width * omp_output_image.height * sizeof(unsigned int));

    int i = 0;
#pragma omp parallel for
    for (i = 0; i < omp_particles_count; ++i) {
        int x_min = (int)roundf(omp_particles[i].location[0] - omp_particles[i].radius);
        int y_min = (int)roundf(omp_particles[i].location[1] - omp_particles[i].radius);
        int x_max = (int)roundf(omp_particles[i].location[0] + omp_particles[i].radius);
        int y_max = (int)roundf(omp_particles[i].location[1] + omp_particles[i].radius);

        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= omp_output_image.width ? omp_output_image.width - 1 : x_max;
        y_max = y_max >= omp_output_image.height ? omp_output_image.height - 1 : y_max;

        // For each pixel in the bounding box, check that it falls within the radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - omp_particles[i].location[0];
                const float y_ab = (float)y + 0.5f - omp_particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= omp_particles[i].radius) {
                    const unsigned int pixel_offset = y * omp_output_image.width + x;
#pragma omp atomic
                    ++omp_pixel_particle_count_histogram[pixel_offset];
                }
            }
        }
    }

#ifdef VALIDATION
    validate_pixel_contribs(omp_particles, omp_particles_count, omp_pixel_particle_count_histogram, omp_output_image.width, omp_output_image.height);
#endif
}

void openmp_stage2()
{
    // Exclusive prefix sum across the histogram to create an index
    omp_pixel_index[0] = 0;
    int i = 0;
    for (i = 0; i < omp_output_image.width * omp_output_image.height; ++i)
    {
        omp_pixel_index[i + 1] = omp_pixel_index[i] + omp_pixel_particle_count_histogram[i];
    }

    // Recover the total from the index
    const unsigned int TOTAL_CONTRIBS = omp_pixel_index[omp_output_image.width * omp_output_image.height];

    if (TOTAL_CONTRIBS > omp_pixel_contrib_count) {
        // (Re)Allocate colour storage
        if (omp_pixel_contrib_colours) free(omp_pixel_contrib_colours);
        if (omp_pixel_contrib_depth) free(omp_pixel_contrib_depth);
        omp_pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        omp_pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        omp_pixel_contrib_count = TOTAL_CONTRIBS;
    }

    memset(omp_pixel_particle_count_histogram, 0, omp_output_image.width * omp_output_image.height * sizeof(unsigned int));

    i = 0;
    int x = 0;
#pragma omp parallel for ordered private(x)
    for (i = 0; i < omp_particles_count; ++i)
    {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(omp_particles[i].location[0] - omp_particles[i].radius);
        int y_min = (int)roundf(omp_particles[i].location[1] - omp_particles[i].radius);
        int x_max = (int)roundf(omp_particles[i].location[0] + omp_particles[i].radius);
        int y_max = (int)roundf(omp_particles[i].location[1] + omp_particles[i].radius);

        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= omp_output_image.width ? omp_output_image.width - 1 : x_max;
        y_max = y_max >= omp_output_image.height ? omp_output_image.height - 1 : y_max;

#pragma omp ordered
        {
            x = 0;
            // Store data for every pixel within the bounding box that falls within the radius
            for (x = x_min; x <= x_max; ++x) {
                for (int y = y_min; y <= y_max; ++y) {
                    const float x_ab = (float)x + 0.5f - omp_particles[i].location[0];
                    const float y_ab = (float)y + 0.5f - omp_particles[i].location[1];
                    const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                    if (pixel_distance <= omp_particles[i].radius) {
                        const unsigned int pixel_offset = y * omp_output_image.width + x;

                        const unsigned int storage_offset = omp_pixel_index[pixel_offset] + (omp_pixel_particle_count_histogram[pixel_offset]);
                        // Copy data to cpu_pixel_contrib buffers
                        omp_pixel_particle_count_histogram[pixel_offset]++;
                        memcpy(omp_pixel_contrib_colours + (4 * storage_offset), omp_particles[i].color, 4 * sizeof(unsigned char));
                        memcpy(omp_pixel_contrib_depth + storage_offset, &omp_particles[i].location[2], sizeof(float));
                    }
                }
            }
        }
    }

    // Pair sort the colours contributing to each pixel based on ascending depth
#pragma omp parallel for schedule(guided, 4)
    for (i = 0; i < omp_output_image.width * omp_output_image.height; ++i) {
        cpu_sort_pairs(
            omp_pixel_contrib_depth,
            omp_pixel_contrib_colours,
            omp_pixel_index[i],
            omp_pixel_index[i + 1] - 1
        );
    }

#ifdef VALIDATION
    validate_pixel_index(omp_pixel_particle_count_histogram, omp_pixel_index, omp_output_image.width, omp_output_image.height);
    validate_sorted_pairs(omp_particles, omp_particles_count, omp_pixel_index, omp_output_image.width, omp_output_image.height, omp_pixel_contrib_colours, omp_pixel_contrib_depth);
#endif    
}
void openmp_stage3()
{
    memset(omp_output_image.data, 255, omp_output_image.width * omp_output_image.height * omp_output_image.channels * sizeof(unsigned char));
    int row = 0;
    int col = 0;

#pragma omp parallel for default(none) private(row, col) shared(omp_pixel_index, omp_output_image, omp_pixel_contrib_colours) schedule(static, 2)
    for (row = 0; row < omp_output_image.width * omp_output_image.height; ++row) {
        for (col = omp_pixel_index[row]; col < omp_pixel_index[row + 1]; ++col) {
            const float opacity = (float)omp_pixel_contrib_colours[col * 4 + 3] / (float)255;
            omp_output_image.data[(row * 3) + 0] = (unsigned char)((float)omp_pixel_contrib_colours[col * 4 + 0] * opacity + (float)omp_output_image.data[(row * 3) + 0] * (1 - opacity));
            omp_output_image.data[(row * 3) + 1] = (unsigned char)((float)omp_pixel_contrib_colours[col * 4 + 1] * opacity + (float)omp_output_image.data[(row * 3) + 1] * (1 - opacity));
            omp_output_image.data[(row * 3) + 2] = (unsigned char)((float)omp_pixel_contrib_colours[col * 4 + 2] * opacity + (float)omp_output_image.data[(row * 3) + 2] * (1 - opacity));
        }
    }

#ifdef VALIDATION
    validate_blend(omp_pixel_index, omp_pixel_contrib_colours, &omp_output_image);
#endif    
}
void openmp_end(CImage* output_image)
{
    // Store reReachingturn value
    output_image->width = omp_output_image.width;
    output_image->height = omp_output_image.height;
    output_image->channels = omp_output_image.channels;
    memcpy(output_image->data, omp_output_image.data, omp_output_image.width * omp_output_image.height * omp_output_image.channels * sizeof(unsigned char));

    // Release allocations
    free(omp_pixel_contrib_depth);
    free(omp_pixel_contrib_colours);
    free(omp_output_image.data);
    free(omp_pixel_index);
    free(omp_pixel_particle_count_histogram);
    free(omp_particles);

    // Return ptrs to nullptr
    omp_pixel_contrib_depth = 0;
    omp_pixel_contrib_colours = 0;
    omp_output_image.data = 0;
    omp_pixel_index = 0;
    omp_pixel_particle_count_histogram = 0;
    omp_particles = 0;
}