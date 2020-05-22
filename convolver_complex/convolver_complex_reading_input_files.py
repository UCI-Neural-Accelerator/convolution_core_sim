import random
import sys
import math
import numpy as np


def generate_files_comparison():
    """
    Generates testing data for fixed point convolution testing.

    Args:
        kernel: uint16 numpy array that contains the kernel infomation
        bias: uint16 value for the bias
    """

    # open and read the files
    # image_file = open('input_pixels.txt', 'r')
    # kernel_file = open('input_weights.txt', 'r')
    # bias_file = open('input_bias.txt', 'r')
    output_file = open('output_convolver_py.txt', 'w')
    
    with open('input_pixels.txt', 'r') as image_file:
        tmp_image = [int(x,2) for x in image_file.read().split()]

    with open('input_weights.txt', 'r') as kernel_file:
        tmp_kernel = [int(y,2) for y in kernel_file.read().split()]

    with open('input_bias.txt', 'r') as bias_file:
        bias = [int(z,2) for z in bias_file.read().split()]        
    
    image_size = (28, 28)
    kernel_size = (5, 5)
    
    image = np.zeros(image_size, dtype=np.int16)
    kernel = np.zeros(kernel_size, dtype=np.int16)

    for i in range(28):
        for j in range(28):
            image[i][j] = tmp_image[i*28 + j]

    for i in range(5):
        for j in range(5):
            kernel[i][j] = tmp_kernel[i*5 + j]
    
    #FOR VERIFICATION
    # print(type(tmp_image[1])) #it is an string
    # print(tmp_image[2])
    # print(tmp_image[2*28 + 1])
    # print(tmp_kernel[2])
    # print(type(image[5][6]))
    # print(image[5][6]) #correct one should be 256
    # print(kernel[2][3]) #correct one should be 256
    # print(bias)
    
    # generate and write the output
    output = convolution(image, kernel, bias)

    for pixel in np.nditer(output.flatten()):
        output_file.write(np.binary_repr(pixel, width=16) + '\n') #bin(pixel)

    # close all files
    image_file.close()
    kernel_file.close()
    bias_file.close()
    output_file.close()


def convolution(image: np.ndarray, kernel: np.ndarray, bias: np.uint16) -> np.ndarray:
    """
    Computes convolution with input image and kernel
    Args:
        image: uint16 numpy array 
        kernel: uint16 numpy array
        bias: uint16 value
         
    Returns:
        output image
    """
    
    # calculate the border thickness from convolution
    border_thickness = (math.floor(kernel.shape[0] / 2), math.floor(kernel.shape[1] / 2))

    # calculate the size of the output image
    output_size = (image.shape[0] - (2 * border_thickness[0]), image.shape[1] - (2 * border_thickness[0]))

    # create the empty output image array
    output = np.zeros(output_size, dtype=np.int16)

    # iterate over input image
    for x in range(output_size[0]):
        for y in range(output_size[1]):
            # multiplication
            mult = np.zeros((kernel.shape[0], kernel.shape[1]), dtype=np.int16)
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    mult[i][j] = mult_fixed_point(kernel[i][j], image[x + i][y + j])

            # accumulation
            sum = np.sum(mult, dtype=np.int16)

            # bias
            sum = add_fixed_point(sum, bias)

            output[x][y] = sum

    return output


def mult_fixed_point(pixel: np.int16, weight: np.int16) -> np.int16:

    # multiply the inputs
    mult = np.int32(pixel) * np.int32(weight)

    # return fixed point result
    return np.int16(mult >> 8)


def add_fixed_point(a: np.int16, b: np.int16) -> np.int16:
    
    # return fixed point result
    return np.int16(a + b)


if __name__ == '__main__':
    generate_files_comparison()
