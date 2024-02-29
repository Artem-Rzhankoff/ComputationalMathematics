#include <stddef.h>
#include "helper.h"

double randfrom(double min, double max);

void random_padding(double** matrix, size_t size);

double** create_matrix(size_t size, func u, double h);

double** create_f_matrix(size_t size, func f, double h);

env create_env(size_t size, func f, func u);
