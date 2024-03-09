#include <stddef.h>
#include <math.h>
#include "utils.h"
#include "helper.h"

void calculate_aproxy_gauss(double** matrix, double** f, size_t size)
{
    double h = 1.0 / (size + 1), dmax = 0;
    int i, j;
    double dm, temp;

    do {
        dmax = 0;
        for (i = 1; i < size + 1; ++i) {
            for (j = 1; j < size + 1; ++j) {
                temp = matrix[i][j];
                matrix[i][j] = 0.25 * (matrix[i-1][j] + matrix[i+1][j] + 
                    matrix[i][j-1] + matrix[i][j+1] - h * h * f[i][j]);

                dm = fabs(temp - matrix[i][j]);
                if (dmax - dm) dmax = dm;
            }
        }
    } while (dmax > EPSILON);
}