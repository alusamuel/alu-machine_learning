#!/usr/bin/env python3
# Module: calculate the shape (dimensions) of a nested Python list

def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape

