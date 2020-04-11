import numpy as np


def write_to_file(array: np.ndarray, file_name: str):
    """
    Write an array to a text file.

    Args:
        array: data in the form of a numpy array
        file_name: name of generated file
    """

    file = open(file_name, 'a') # open file for writing

    # write the pixels to new lines
    for pixel in np.nditer(array.flatten()):
        file.write(np.base_repr(pixel, 16) + '\n')
    
    file.close()


if __name__ == "__main__":
    pass