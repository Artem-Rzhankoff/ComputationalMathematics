#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "helper.h"

typedef struct {
    double average;
    double max;
} error;

double randfrom(double min, double max)
{
    double range = max - min;
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void random_padding1(double** matrix, size_t size)
{
    for (int i = 1; i < size + 1; ++i)
        for (int j = 1; j < size + 1; ++j)
            matrix[i][j] = 0;//randfrom(-100.0, 100.0);
}

void random_padding(double** matrix, size_t size)
{
    int n = 0;
    double mod = 0;
    for (int i = 0; i < size + 2; ++i) {
        n += 1;
        mod = (mod * (n - 1) + matrix[i][0]) / n;
        n += 1;
        mod = (mod * (n - 1) + matrix[0][i]) / n;
        n += 1;
        mod = (mod * (n - 1) + matrix[size+1][i]) / n;
        n += 1;
        mod = (mod * (n - 1) + matrix[i][size+1]) / n;
    }

    for (int i = 1; i < size + 1; ++i) {
        for (int j = 1; j < size + 1; ++j) {
            matrix[i][j] = mod;
        }
    }
}

void edge_cases_padding(double** matrix, size_t size, func u, double h)
{
    int i;

    for (i = 0; i < size + 2; ++i) {
        matrix[0][i] = u(0, i * h);
        matrix[size+1][i] = u(1, i * h);
        matrix[i][0] = u(i * h, 0);
        matrix[i][size+1] = u(i * h, 1);
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

error calculate_approximation_error(func u, env my_env) 
{
    double** approximate_u = my_env.u;
    size_t size = my_env.size;
    double h = my_env.h;
    double sum = 0, max = 0;
    for (int i = 1; i < size + 1; ++i) {
        for (int j = 1; j < size + 1; ++j) {
            double u_ij = u(i * h, j * h);
            if (u_ij > max) {
                max = u_ij;
            }
            if (u_ij == 0) {
                //sum += fabs((approximate_u[i][j] - u_ij) / 1e-6);
            } else {
                sum += fabs(approximate_u[i][j] - u_ij) / u_ij;
            }
        }
    }
    error e;
    e.average = sum / pow(size, 2);
    e.max = max;

    return e;
}