#include <omp.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "utils.h"
#include "multithread_wave.h"

#define THREADS_AMOUNT omp_get_num_procs()

double base_f(double x, double y) { return 0; }
double base_u(double x, double y)
{
    if (y == 0)
        return 100 - 200 * x;
    else if (x == 0)
        return 100 - 200 * y;
    else if (y == 1)
        return -100 + 200 * x;
    else if (x == 1)
        return -100 + 200 * y;

    exit(-1);
}

double ftc_f(double x, double y) { return 6000 * x + 12000 * y; }
double ftc_u(double x, double y) { return 1000 * pow(x, 3) + 2000 * pow(y, 3); }

// TESTS

void first_test_case(size_t size)
{
    env my_env = create_env(size, base_f, base_u);
    double** matrix = create_matrix(size, base_u, my_env.h);

    // однопоточная

    double st_start = omp_get_wtime();
    calculate_aproxy(matrix, my_env.f, my_env.size);
    double st_end = omp_get_wtime();

    printf("| %.3f ", st_end - st_start);

    // многопоточная

    omp_set_num_threads(THREADS_AMOUNT);

    double mt_start = omp_get_wtime();
    calculate_aproxy(my_env.u, my_env.f, my_env.size);
    double mt_end = omp_get_wtime();

    omp_set_num_threads(1);

    printf("| %.3f |\n", mt_end - mt_start);

}

int main() 
{
    size_t sz[6] = {100, 300, 500, 700, 900, 2000};

    for (int i = 0; i < 6; ++i) {
        first_test_case(sz[i]);
    }

    return 0;
}