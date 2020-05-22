import random
import sys
import math
import numpy as np

#import file_out


def generate_random_data(samples=1):
    """
    Generates random data for the number of sample inputs wanted
    
    Args:
        samples: int value for the number of inputs that will be tested
    """

    # generate multiple samples
    for i in range(samples):
        image = rand_image(28, 28)
        kernel = rand_kernel(5)
        bias = rand_bias()

    generate_files(image, kernel, bias)


def generate_files(image: np.ndarray, kernel: np.ndarray, bias: np.int16):
    """
    Generates testing data for fixed point convolution testing.

    Args:
        kernel: uint16 numpy array that contains the kernel infomation
        bias: uint16 value for the bias
    """

    # open the files
    image_file = open('input_pixels.txt', 'w')
    kernel_file = open('input_weights.txt', 'w')
    bias_file = open('input_bias.txt', 'w')
    output_file = open('output_convolver_py.txt', 'w')

    # write the pixels to new lines
    for pixel in np.nditer(image.flatten()):
        image_file.write(np.binary_repr(pixel, width=16) + '\n')

    # write the kernel values to new lines
    for value in np.nditer(kernel.flatten()):
        kernel_file.write(np.binary_repr(value, width=16) + '\n')

    # write the bias to new line
    bias_file.write(np.binary_repr(bias, width=16) + '\n')
    
    # generate and write the output
    output = convolution(image, kernel, bias)

    for pixel in np.nditer(output.flatten()):
        output_file.write(np.binary_repr(pixel, width=16) + '\n')

    # close all files
    image_file.close()
    kernel_file.close()
    bias_file.close()
    output_file.close()


def rand_image(height: int, width: int) -> np.ndarray:
    """
    Generates a random test image

    Args:
        height: height of test image
        width: width of test image
    
    Returns:
        test image as numpy array
    """

    # generate random image with correct datatype
    image = (np.random.rand(height, width) * 65535).astype(np.uint16)

    return image
    



def rand_kernel(width: int) -> np.ndarray:
    """
    Generates a random kernel
    Args:
        width: width of square kernel
        
    Returns:
        kernel as numpy array
    """

    # generate random kernel size width*width with correct datatype
    kernel = (np.random.rand(width, width) * 65535).astype(np.uint16)
    
    return kernel

    


def rand_bias():
    """
    Generates bias

    Returns:
        randomly generated bias 
    
    """
    #create empty bias as 16 bit unsigned int
    bias = (np.random.rand(1) * 65535).astype(np.uint16)[0]
    
    return bias


def convolution(image: np.ndarray, kernel: np.ndarray, bias: np.int16) -> np.ndarray:
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
    output_size = (image.shape[0] - (2 * border_thickness[0]), image.shape[1] - (2 * border_thickness[1]))

    # create the empty output image array
    output = np.zeros(output_size, dtype=np.int16)

    # iterate over input image
    for x in range(output_size[0]):
        for y in range(output_size[1]):
            # multiplication
            mult = np.zeros((kernel.shape[0], kernel.shape[1]), dtype=np.int16)
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    mult[i, j] = mult_fixed_point(kernel[i, j], image[x + i, y + j])

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

# def return_fixed_point(a: np.uint16) -> str:

    # integer_part = np.binary_repr(np.uint8(a >> 8))
    # frac_part = np.binary_repr(np.uint8(a))

    # return integer_part + '.' + frac_part


if __name__ == '__main__':
    generate_random_data()
