#define EPSILON 0.01
#define BLOCK_SIZE 200 // в зависимости от размера кэша или чего там бля

typedef struct Edge_cases {
    double zero_x[2]; // в массивах представлены коэфициенты уравнения
    double zero_y[2];
    double one_x[2];
    double one_y[2];
} edge_cases;