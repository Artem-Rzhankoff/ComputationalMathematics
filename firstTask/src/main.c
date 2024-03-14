#include <omp.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "gauss_seidel.h"
#include "utils.h"
#include "multithread_wave.h"

#define THREADS_AMOUNT omp_get_num_procs()

double ftc_f(double x, double y) { return 0; }
double ftc_u(double x, double y)
{
    if (y == 0)
        return 100 - 200 * x;
    else if (x == 0)
        return 100 - 200 * y;
    else if (y == 1)
        return -100 + 200 * x;
    else if (x == 1)
        return -100 + 200 * y;
    else
        return 0;
}

double stc_f(double x, double y) { return 6000 * pow(x, 2) + 9000 * y; }
double stc_u(double x, double y) { return 500 * pow(x, 4) + 1500 * pow(y, 3); }

double ttc_f(double x, double y) { return -sin(x) / 10 - cos(y) / 10; }
double ttc_u(double x, double y) { return sin(10 * x) / 10 + cos(10 * y) / 10; }

double thtc_f(double x, double y) { return 10 / pow(2*x+y+1/100, 3); }
double thtc_u(double x, double y) { return 1 / (pow(x, 2) + pow(y, 2) + 0.005); }

// TESTS

int unambiguity_single_multi_threading_versions() {
    // аппроксимация, полученная однопоточной и мультипоточной версиями,должны совпадать
    
    size_t size = 700;
    env my_env = create_env(size, ftc_f, ftc_u);
    double** matrix = allocate_memory(size+2);

    for (int i = 0; i < size+2; ++i) {
        for (int j = 0; j < size+2; ++j) {
            matrix[i][j] = my_env.u[i][j];
        }
    }

    calculate_aproxy(matrix, my_env.f, my_env.size);

    omp_set_num_threads(THREADS_AMOUNT);
    calculate_aproxy(my_env.u, my_env.f, my_env.size);
    omp_set_num_threads(1);

    int flag = 0;
    for (int i = 0; i < size+2; ++i) {
        for (int j = 0; j < size+2; ++j) {
            if (fabs(matrix[i][j] - my_env.u[i][j]) > 1e-6) {
                printf("i=%d, j=%d, matrix=%f, my.u=%f\n", i, j, matrix[i][j], my_env.u[i][j]);
                return -1;
            }
        }
    }

    return 0;
}


void first_test(size_t size, int threads_amount, FILE* fptr) {
    env my_env = create_env(size, ftc_f, ftc_u);

    omp_set_num_threads(threads_amount);

    double mt_start = omp_get_wtime();
    calculate_aproxy(my_env.u, my_env.f, my_env.size);
    double mt_end = omp_get_wtime();

    fprintf(fptr, "%.4f ", mt_end - mt_start);
}

void run_test(size_t size, int threads_amount, FILE* fptr, func f, func u) 
{
    env my_env = create_env(size, f, u);

    omp_set_num_threads(threads_amount);
    double mt_start = omp_get_wtime();

    calculate_aproxy(my_env.u, my_env.f, my_env.size);

    double mt_end = omp_get_wtime();
    omp_set_num_threads(1);

    fprintf(fptr, "%.4f ", fabs(mt_end - mt_start));

    if (threads_amount > 1) {
        error e = calculate_approximation_error(f, my_env);
        fprintf(fptr, "%.5f ", e.average);
    }
}

int main() 
{
    if (unambiguity_single_multi_threading_versions() != 0) {
        return -1;
    }

    FILE *fptr = fopen("../result1.txt", "w");
    size_t sz[6] = {200, 300, 500, 700, 900, 2000};

    for (int i = 0; i < 6; ++i) {
        fprintf(fptr, "%d ", sz[i]);
        first_test(sz[i], 1, fptr);
        first_test(sz[i], THREADS_AMOUNT, fptr);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    fptr = fopen("../result2.txt", "w");
    for (int i = 0; i < 6; ++i) {
        fprintf(fptr, "%d ", sz[i]);
        run_test(sz[i], 1, fptr, stc_f, stc_u);
        run_test(sz[i], THREADS_AMOUNT, fptr, stc_f, stc_u);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    fptr = fopen("../result3.txt", "w");
    for (int i = 0; i < 6; ++i) {
        fprintf(fptr, "%d ", sz[i]);
        run_test(sz[i], 1, fptr, ttc_f, ttc_u);
        run_test(sz[i], THREADS_AMOUNT, fptr, ttc_f, ttc_u);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    fptr = fopen("../result4.txt", "w");
    for (int i = 0; i < 6; ++i) {
        fprintf(fptr, "%d ", sz[i]);
        run_test(sz[i], 1, fptr, ttc_f, ttc_u);
        run_test(sz[i], THREADS_AMOUNT, fptr, ttc_f, ttc_u);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    return 0;
}