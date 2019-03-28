#include "utils.h"

#include <stdio.h>

__global__
void calcAnswerMapKernel(
        const unsigned char * const inputImage,
        double * const answerImage,
        int numXInput,
        int numYInput,
        int numZInput,
        double* const AMat,
        int AMatLength
        ) {

    int patchRadius = 5;
    // TODO types beim erstellen des features

    int absolute_position_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int absolute_position_y = (blockIdx.y * blockDim.y) + threadIdx.y;


    int x = absolute_position_x - patchRadius;
    int y = absolute_position_y - patchRadius;
    int w = absolute_position_x + patchRadius;
    int h = absolute_position_y + patchRadius;

    if (x < 0 || y < 0 || w > numXInput || h > numYInput) {
        //        printf("Returning %i | %i\n", absolute_position_x, absolute_position_y);
        return;
    }


    unsigned char rgba = inputImage[absolute_position_y + absolute_position_x * numYInput];
    double channelSum = .299f * rgba;
    answerImage[absolute_position_y + absolute_position_x * numYInput] = channelSum;

    //    printf("%i | %i \n", absolute_position_x, absolute_position_y);

}

void startCalcAnswerMap(
        unsigned char * const d_inputImage,
        double * const d_answerImage,
        size_t numXInput,
        size_t numYInput,
        size_t numZInput,
        double* const d_AMat,
        int AMatLength
        ) {
    int blocksize = 32;
    const dim3 blockSize(blocksize, blocksize, 1);
    const dim3 gridSize(ceil(numXInput / (double) blocksize), ceil(numYInput / (double) blocksize), 1);

    std::cout << "SIZE: " << numXInput << " " << numYInput << std::endl;
    std::cout << "blocksize: " << blockSize.x << " " << blockSize.y << std::endl;
    std::cout << "gridSize: " << gridSize.x << " " << gridSize.y << std::endl;

    calcAnswerMapKernel << <gridSize, blockSize >> >(d_inputImage, d_answerImage, numXInput, numYInput, numZInput, d_AMat, AMatLength);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}