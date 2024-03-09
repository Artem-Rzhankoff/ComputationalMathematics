#include <omp.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

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

double stc_f(double x, double y) { return 6000 * x + 12000 * y; }
double stc_u(double x, double y) { return 1000 * pow(x, 3) + 2000 * pow(y, 3); }

//double f_sin(double x, double y) { return sin(x) + 0 * y; }


double ttc_f(double x, double y) { return -sin(x) / 10000 - sin(y) / 10000; }
double ttc_u(double x, double y) { return sin(x) / 10000 + sin(y) / 10000; }

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
                printf("i=%d, j=%d, matrix=%f, my.u=%f", i, j, matrix[i][j], my_env.u[i][j]);
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

    fprintf(fptr, "     %8.4fs    |", mt_end - mt_start);
}

void second_test(size_t size, int threads_amount, FILE* fptr) {
    env my_env = create_env(size, stc_f, stc_u);

    omp_set_num_threads(threads_amount);

    double mt_start = omp_get_wtime();
    calculate_aproxy(my_env.u, my_env.f, my_env.size);
    double mt_end = omp_get_wtime();

    fprintf(fptr, "     %8.4fs    |", mt_end - mt_start);

    if (threads_amount > 1) {
        error e = calculate_approximation_error(stc_u, my_env);
        fprintf(fptr, "     %10.5f     |", e.average);
        //fprintf(fptr, "     %10.5f     |", e.max);
    }
}

void third_test(size_t size, int threads_amount, FILE* fptr) {
    env my_env = create_env(size, ttc_f, ttc_u);

    omp_set_num_threads(threads_amount);

    double mt_start = omp_get_wtime();
    calculate_aproxy(my_env.u, my_env.f, my_env.size);
    double mt_end = omp_get_wtime();

    fprintf(fptr, "     %8.4fs    |", mt_end - mt_start);

    if (threads_amount > 1) {
        error e = calculate_approximation_error(ttc_u, my_env);
        fprintf(fptr, "     %10.5f     |", e.average);
        //fprintf(fptr, "     %10.5f     |", e.max);
    }
}

int main() 
{
    if (unambiguity_single_multi_threading_versions() != 0) {
        return -1;
    }

    FILE *fptr = fopen("results.txt", "w");
    size_t sz[6] = {200, 300, 500, 700, 900, 2000};

    fprintf(fptr, "#################TEST 1#################\n");
    fprintf(fptr, " |  GRID SIZE  | TIME  (1 THREAD) | TIME (12 THREADS) |\n");
    fprintf(fptr, " *-------------*-----------------*-------------------*\n");
    for (int i = 0; i < 4; ++i) {
        fprintf(fptr, " |    %.4d     |", sz[i]);
        first_test(sz[i], 1, fptr);
        first_test(sz[i], THREADS_AMOUNT, fptr);
        fprintf(fptr, "\n *-------------*-----------------*------------------*\n");
    }
    fprintf(fptr, "\n");

    fprintf(fptr, "#################TEST 2#################\n");
    fprintf(fptr, " |  GRID SIZE  | TIME  (1 THREAD) | TIME (12 THREADS) | APPROXIMATION ERROR|\n");
    fprintf(fptr, " *-------------*-----------------*-------------------*--------------------*\n");
    for (int i = 0; i < 4; ++i) {
        fprintf(fptr, " |    %.4d     |", sz[i]);
        second_test(sz[i], 1, fptr);
        second_test(sz[i], THREADS_AMOUNT, fptr);
        fprintf(fptr, "\n *-------------*-----------------*-------------------*--------------------*\n");
    }
    fprintf(fptr, "\n");

    fprintf(fptr, "#################TEST 3#################\n");
    fprintf(fptr, " |  GRID SIZE  | TIME  (1 THREAD) | TIME (12 THREADS) | APPROXIMATION ERROR|\n");
    fprintf(fptr, " *-------------*-----------------*-------------------*--------------------*\n");
    for (int i = 0; i < 4; ++i) {
        fprintf(fptr, " |    %.4d     |", sz[i]);
        third_test(sz[i], 1, fptr);
        third_test(sz[i], THREADS_AMOUNT, fptr);
        fprintf(fptr, "\n *-------------*-----------------*-------------------*--------------------*\n");
    }

    return 0;
}