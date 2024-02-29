#define EPSILON 0.01
#define BLOCK_SIZE 200

typedef double (*func)(double, double);

typedef struct {
    size_t size;
    double** u;
    double** f;
    double h;
} env;