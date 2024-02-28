#include <stddef.h>
#include <omp.h>
#include "helper.h"


double calculate_block(double** matrix, size_t size, int x_start, int y_start, int h, double** f)
{
    int x_start = x_start * BLOCK_SIZE, y_start = y_start * BLOCK_SIZE;
    int x_finish = x_start + fmin(BLOCK_SIZE, size - x_start);
    int y_finish = y_start + fmin(BLOCK_SIZE, size - y_start);

    double dm = 0;

    for (int i = x_start + 1; i < x_finish + 1; ++i) {
        for (int j = y_start + 1; j < y_finish + 1; ++j) {
            double temp = matrix[i][j];
            matrix[i][j] = 0.25 * (matrix[i-1][j] + matrix[i+1][j] + 
                    matrix[i][j-1] + matrix[i][j+1] - h * h * f[i][j]);

            double d = fabs(temp - matrix[i][j]);
            if (dm < d) dm = d;
        }
    }

    return dm;
}

void calculate_aproxy(double** matrix, double** f ,size_t size) 
{
    double h = 1 / (size + 1), dmax = 0;
    int i, j;
    double* dm = malloc (size * sizeof(double));
    size_t block_amount = size / BLOCK_SIZE; // посмотреть в какую сторону округлять и надо ли вообще

    omp_lock_t dmax_lock;
    omp_init_lock (&dmax_lock);
    do {
        dmax = 0;
        for (int wavelength = 0; wavelength < block_amount; ++wavelength) { // размер волны = wavelength + 1
            dm[wavelength] = 0;
#pragma omp parallel for shared(matrix, wavelength, dm) private(i, j, d)
            for (i=0; i < wavelength + 1; ++i) {
                j = wavelength - i;
                // тут обрабатываем блок-блок бейби
                double d = calculate_block(matrix, size, i, j, h, f);
                if (dm[i] < d) dm[i] = d;
            }
        }

        for (int wavelength = block_amount - 2; wavelength > -1; --wavelength) {
#pragma omp parallel for shared(matrix, wavelength, dm) private(i, j, temp, dm)
            for (i = 0; i < wavelength + 1; ++i) {
                j = 2 * (block_amount - 1) - wavelength - i;

                double d = calculate_block(matrix, size, i, j, h, f);
                if (dm[i] < d) dm[i] = d;
            }            
        } // size -> wavelength ??
#pragma omp parallel for shared(size, dm, dmax)\
                         private(i)
        for (i = 1; i < block_amount + 1; ++i) {
            omp_set_lock(&dmax_lock);
                if (dm[i] > dmax) dmax = dm[i];
            omp_unset_lock(&dmax_lock); 
        }

    } while (dmax > EPSILON);
}