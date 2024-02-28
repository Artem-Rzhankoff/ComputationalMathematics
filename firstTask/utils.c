#include <stdlib.h>
#include <stddef.h>
#include "helper.h"

double randfrom(double min, double max)
{
    double range = max - min;
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void random_padding(double** matrix, size_t size)
{
    for (int i = 1; i < size + 1; ++i)
        for (int j = 1; j < size + 1; ++j)
            matrix[i][j] = randfrom(-100.0, 100.0);
}

void edge_cases_padding(double** matrix, size_t size, edge_cases edge)
{
    int i;
    for (i = 0; i < size + 2; ++i) {
        matrix[0][i] = edge.zero_x[0] + edge.zero_x[1] * i;
        matrix[size][i] = edge.one_x[0] + edge.one_x[1] * i;
    }
    for (i = 0; i < size + 2; ++i) {
        matrix[i][0] = edge.zero_y[0] + edge.zero_y[1] * i;
        matrix[i][size] = edge.one_y[0] + edge.one_y[1] * i;
    }
}

double** create_matrix(size_t size, edge_cases edge)
{
    double* matrix[size];
    for (int i = 0; i < size + 2; ++i) {
        matrix[i] = (double*)malloc(size * sizeof(double));
    }
    // тут наверное можно заполнение ебануть
    edge_cases_padding(matrix, size, edge);

    random_padding(matrix, size);

    return matrix;
}