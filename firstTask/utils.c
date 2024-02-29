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

void edge_cases_padding(double** matrix, size_t size, func u, double h)
{
    int i;

    for (i = 0; i < size + 2; ++i) {
        matrix[0][i] = u(0, i * h);
        matrix[size][i] = u(1, i * h);
        matrix[i][0] = u(i * h, 0);
        matrix[i][size] = u(i * h, 1);
    }
}

double** allocate_memory(size_t size)
{
    double** matrix = malloc(size * sizeof(double*));
    for (int i = 0; i < size; ++i) {
        matrix[i] = (double*)malloc(size * sizeof(double));
    }

    return matrix;
}

double** create_matrix(size_t size, func u, double h)
{
    double** matrix = allocate_memory(size + 2);
    edge_cases_padding(matrix, size, u, h);
    random_padding(matrix, size);

    return matrix;
}

double** create_f_matrix(size_t size, func f, double h)
{
    double** matrix = allocate_memory(size + 2);
    for (int i = 0; i < size + 2; ++i) {
        for (int j = 0; j < size + 2; ++j) {
            matrix[i][j] = f(i * h, j * h);
        }
    }

    return matrix;
}

env create_env(size_t size, func f, func u)
{
    env my_env;
    double h = 1.0 / (size + 1);

    my_env.size = size;
    my_env.f = create_f_matrix(size, f, h);
    my_env.u = create_matrix(size, u, h);
    my_env.h = h;

    return my_env;
}