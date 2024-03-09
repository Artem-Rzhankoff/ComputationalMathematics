#include <stddef.h>
#include "helper.h"

typedef struct {
    double average;
    double max;
} error;

double** allocate_memory(size_t size);

double** create_matrix(size_t size, func u, double h);

double** create_f_matrix(size_t size, func f, double h);

env create_env(size_t size, func f, func u);

error calculate_approximation_error(func u, env my_env);
