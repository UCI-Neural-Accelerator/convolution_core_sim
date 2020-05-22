import random
import sys
import math
import numpy as np


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

    # open the files
    image_file = open('input_pixels.txt', 'w')
    kernel_file = open('input_weights.txt', 'w')
    bias_file = open('input_bias.txt', 'w')

    # write the pixels to new lines
    for pixel in np.nditer(image.flatten()):
        image_file.write(np.binary_repr(pixel, width=16) + '\n')
        #image_file.write('000000' + np.binary_repr(pixel, width=3) + '0000000' + '\n') #FOR SMALL VALUES

    # write the kernel values to new lines
    for value in np.nditer(kernel.flatten()):
        kernel_file.write(np.binary_repr(value, width=16) + '\n')        
        #kernel_file.write('000000' + np.binary_repr(value, width=3) + '0000000' + '\n') #FOR SMALL VALUES

    # write the bias to new line
    bias_file.write(np.binary_repr(bias, width=16) + '\n')    
    #bias_file.write('000000' + np.binary_repr(bias, width=3) + '0000000' + '\n') #FOR SMALL VALUES
    
    # close all files
    image_file.close()
    kernel_file.close()
    bias_file.close()


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
    image = (np.random.rand(height, width) * 65535).astype(np.uint16) #np.random.rand returns value between 0 and 1

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


# def return_fixed_point(a: np.uint16) -> str:

    # integer_part = np.binary_repr(np.uint8(a >> 8))
    # frac_part = np.binary_repr(np.uint8(a))

    # return integer_part + '.' + frac_part


if __name__ == '__main__':
    generate_random_data()
