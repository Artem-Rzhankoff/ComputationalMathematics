#include <omp.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>


#define EPSILON 0.01

// единичный квадрат -- область D
// нужна функция для определения N - количества узлов

// завести представление граничных случаев
typedef struct Edge_cases {
    double zero_x[2]; // в массивах представлены коэфициенты уравнения
    double zero_y[2];
    double one_x[2];
    double one_y[2];
} edge_cases;

double randfrom(double min, double max)
{
    double range = max - min;
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void random_padding(double** matrix, size_t size)
{
    for (int i = 1; i < size; ++i)
        for (int j = 1; j < size; ++j)
            matrix[i][j] = randfrom(-100.0, 100.0);
}

void edge_cases_padding(double** matrix, size_t size, edge_cases edge)
{
    int i;
    for (i = 0; i < size + 1; ++i) {
        matrix[0][i] = edge.zero_x[0] + edge.zero_x[1] * i;
        matrix[size][i] = edge.one_x[0] + edge.one_x[1] * i;
    }
    for (i = 0; i < size + 1; ++i) {
        matrix[i][0] = edge.zero_y[0] + edge.zero_y[1] * i;
        matrix[i][size] = edge.one_y[0] + edge.one_y[1] * i;
    }
}

// выделить память для массива
double** create_matrix(size_t size, edge_cases edge) 
{
    double* matrix[size];
    for (int i = 0; i < size; ++i) {
        matrix[i] = (double*)malloc(size * sizeof(double));
    }
    // тут наверное можно заполнение ебануть
    edge_cases_padding(matrix, size, edge);

    random_padding(matrix, size);

    return matrix;
}
// у нас n+1 точка, следовательно заполняем n-1

void calculate_aproxy(double** matrix, double** f ,size_t size) 
{
    double h = 1 / (size + 1), dmax = 0;
    int i, j;

    omp_lock_t dmax_lock;
    omp_init_lock (&dmax_lock);
    do {
        dmax = 0;
#pragma omp parallel for shared(matrix, size, dmax)\
                         private(i, j, temp, d, dm)
        for (i = 1; i < size + 1; ++i) { // size ?
            double dm = 0;
            for (j = 1; j < size + 1; ++j) {

                double temp = matrix[i][j];
                matrix[i][j] = 0.25 * (matrix[i-1][j] + matrix[i+1][j] 
                + matrix[i][j-1] + matrix[i][j+1] - h * h * f[i][j]);

                double d = fabs(temp - matrix[i][j]);
                if (dm < d) dm = d;
            }
            omp_set_lock(&dmax_lock);
            if (dmax < dm) dmax = dm;
            omp_unset_lock(&dmax_lock);
        }
    } while (dmax > EPSILON);
}

void main() 
{
    edge_cases edge;
    // число потоков = число процессоров на компе

}