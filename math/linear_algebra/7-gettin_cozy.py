#!/usr/bin/env python3
'''
    A function def cat_matrices2D(mat1, mat2, axis=0):
    that concatenates two matrices along a specific axis:
'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''
        A function def cat_matrices2D(mat1, mat2, axis=0):
        that concatenates two matrices along a specific axis:
    '''
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
