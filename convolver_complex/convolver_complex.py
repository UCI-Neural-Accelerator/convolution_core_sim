import random
import sys
import numpy as np


def generate_test_data(kernel: np.ndarray, bias: np.uint16, samples: int):
    """
    """

    # data to be written
    data = np.empty(1, np.uint16)

    # generate the required number of samples
    for i in range(samples):
        # generate random image
        image = np.random.randint(((2**16)-1), size=(5, 5), dtype=np.uint16)

        # perform convolution operation
        output = convolution(image, kernel, bias)

        # add image and output to data
        data = np.append(data, image)
        data = np.append(data, output)

    data.tofile("test.bin")




def convolution(image: np.ndarray, kernel: np.ndarray, bias: np.uint16, printing=False) -> np.uint16:
    if (printing):
        print("KERNEL:")
        for row in kernel:
            for element in row:
                print(return_fixed_point(element), end='')
                print("\t", end='')
            print('')
        print('')

        print("IMAGE:")
        for row in image:
            for element in row:
                print(return_fixed_point(element), end='')
                print("\t", end='')
            print('')
        print('')
        
        print('BIAS:')
        print(return_fixed_point(bias))
        print('')


    # multiplication
    mult = np.zeros((5, 5), dtype=np.uint16)
    for i in range(5):
        for j in range(5):
            mult[i, j] = mult_fixed_point(kernel[i, j], image[i, j])

    # accumulation
    sum = np.sum(mult, dtype=np.uint16)

    # bias
    sum = add_fixed_point(sum, bias)

    if (printing):
        print('RESULT:')
        print(return_fixed_point(sum))

    return sum



def mult_fixed_point(pixel: np.uint16, weight: np.uint16) -> np.uint16:

    # multiply the inputs
    mult = np.uint32(pixel) * np.uint32(weight)

    # return fixed point result
    return np.uint16(mult >> 8)


def add_fixed_point(a: np.uint16, b: np.uint16) -> np.uint16:
    
    # return fixed point result
    return np.uint16(a + b)

def return_fixed_point(a: np.uint16) -> str:

    integer_part = np.binary_repr(np.uint8(a >> 8))
    frac_part = np.binary_repr(np.uint8(a))

    return integer_part + '.' + frac_part
     

if __name__ == '__main__':
    kernel = np.zeros((5, 5), np.uint16)
    bias = np.zeros(1, np.uint16)
    
    kernel[:, :] = 0x0100
    bias = 0x0000
    
    generate_test_data(kernel, bias, 10)