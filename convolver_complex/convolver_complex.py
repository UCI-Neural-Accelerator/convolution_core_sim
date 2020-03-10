import random
import sys

DATA_WIDTH = 16

BIAS = 0b0000000000000000

KERNEL_SIZE = (5, 5)
KERNEL = [
            [
                (1 << 8) for i in range(KERNEL_SIZE[0])
            ] for i in range(KERNEL_SIZE[1])
        ]
'''KERNEL = [
            [
                random.randint(0, 9) for i in range(KERNEL_SIZE[0])
            ] for i in range(KERNEL_SIZE[1])
        ]'''
IMAGE_SIZE = (5, 5)
IMAGE = [
            [
                # generate random integers of the correct binary size
                (2 << 8) for i in range(IMAGE_SIZE[0])
            ] for i in range(IMAGE_SIZE[1])
        ]
'''IMAGE = [
            [
                # generate random integers of the correct binary size
                random.randint(0, (2 ** 16) - 1) for i in range(IMAGE_SIZE[0])
            ] for i in range(IMAGE_SIZE[1])
        ]'''



def mult_fixed_point(pixel: int, weight: int) -> int:

    # multiply the inputs
    mult = pixel * weight

    # get size of the result
    size = sys.getsizeof(mult)

    # return fixed point result
    return (((mult >> (size - 1)) << (DATA_WIDTH - 1)) | ((mult & 0b011111111111111100000000) >> (int(DATA_WIDTH / 2))))


def add_fixed_point(a: int, b: int) -> int:

    # add the inputs
    add = a + b

    # get the size of the result
    size = sys.getsizeof(add)

    # return fixed point result
    return (((add >> (size - 1)) << (DATA_WIDTH - 1)) | (add & 0b0111111111111111))

def return_fixed_point(a: int) -> str:

    integer_part = bin((a & 0b1111111100000000) >> 8)[2:]
    frac_part = bin(a & 0b0000000011111111)[2:]

    return integer_part + '.' + frac_part
     


def unit():

    print("KERNEL:")
    for row in KERNEL:
        for element in row:
            print(return_fixed_point(element), end='')
            print("\t", end='')
        print('')
    print('')

    print("IMAGE:")
    for row in IMAGE:
        for element in row:
            print(return_fixed_point(element), end='')
            print("\t", end='')
        print('')
    print('')
    
    print('BIAS:')
    print(return_fixed_point(BIAS))
    print('')


    # multiplication
    mult = []
    for i in range(KERNEL_SIZE[0]):
        for j in range(KERNEL_SIZE[1]):
            mult.append(mult_fixed_point(KERNEL[i][j], IMAGE[i][j]))

    # accumulation
    sum = 0
    for element in mult:
        sum = add_fixed_point(sum, element)

    # bias
    sum = add_fixed_point(sum, BIAS)

    print('RESULT:')
    print(return_fixed_point(sum))

if __name__ == '__main__':
    unit()