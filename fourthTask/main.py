import numpy as np
from math import sqrt, cos, sin
import scipy.linalg as la

# алгоритм взят из http://www.ict.nsc.ru/matmod/files/textbooks/KhakimzyanovCherny-2.pdf

# определим базисную функцию
def phi_j(j, xj, x, N):
    h = x[1] - x[0]

    if (j == 0):
        if ((x[0] <= xj) and (xj <= x[1])):
            return (x[1] - xj) / h
        else:
            return 0
    elif (j == N):
        if ((x[N] <= xj) and (x[N-1] >= xj)):
            return (xj - x[N-1]) / h
        else:
            return 0
    else:
        if (x[j-1] <= xj) and (x[j] >= xj):
            return (xj - x[j-1]) / h
        elif (x[j] <= xj) and (x[j+1] >= xj):
            return (x[j+1] - xj) / h
        else:
            return 0

def calc(y, xj, x, N):
    l, r = 0, N-1

    while (r - l > 1):
        m = (l + r) // 2
        if (xj > x[m]):
            l = m
        else:
            r = m

    return y[l] * phi_j(l, xj, x, N) + y[r] * phi_j(r, xj, x, N) 

# K_jk = (phi_k, phi_j)_A
def left_side(lambd, j, k, x):
    h = x[1] - x[0]
    if (j > k):
        j, k = k, j
    
    if (j == k):

        return (x[j+1] + lambd * x[j]**2 * x[j+1] - lambd * x[j] * x[j+1]**2  + (lambd * x[j+1]**3) / 3 - 
                x[j-1] - lambd * x[j]**2 * x[j-1] + lambd * x[j] * x[j-1]**2 - (lambd * x[j-1]**3) / 3) / (h**2)
    elif (j + 1 == k):

        return (-1 / 6.0) * (-6 + lambd * (x[j] - x[j-1])**2) * (x[j] - x[j+1]) / (h**2)
    else:
        return 0

# вычисляем i-ую ячейку вектора в правой части
def right_side(lambd, i, x):
    l_sqrt = sqrt(lambd)
    h = x[1] - x[0]

    a =  2 * (- l_sqrt * (x[i] - x[i+1]) * cos(l_sqrt * x[i]) + sin(l_sqrt * x[i]) - sin(l_sqrt * x[i+1]))
    b = 2 * ( - l_sqrt * (x[i] - x[i-1]) * cos(l_sqrt * x[i]) + sin(l_sqrt * x[i]) - sin(l_sqrt * x[i-1]))

    return (a + b) / h

# метод прогонки
def thomas_algo(a, b, c, d):
    n = len(d)

    for i in range(1, n):
        tmp = a[i] / b[i-1]
        b[i] = b[i] - tmp * c[i-1]
        d[i] = d[i] - tmp * d[i-1]
    
    y = np.zeros(n+1)
    y[n-1] = d[n-1] / b[n-1]

    for i in range(n-2, -1, -1):
        y[i] = (d[i] - c[i] * y[i+1]) / b[i]
    
    return y


def compose_cond_and_solve(lambd, x, N):
    # матрица в левой части
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    # вектор в правой части
    d = np.zeros(N)

    for i in range(1, N+1):
        j = i - 1 # для интервалов
        if (i >= 1):
            a[j] = left_side(lambd, i - 1, i, x)
        if (i < N):
            c[j] = left_side(lambd, i+1, i, x)
        b[j] = left_side(lambd, j, j, x)
        d[j] = right_side(lambd, j, x)

    y = thomas_algo(a, b, c, d)
    # boundary conditions (Дирихле)
    y[0] = 0
    y[N] = 0

    return y



def main():
    lambd = 10
    N = 100

    A, B = 0.0, 4 * np.pi / sqrt(lambd)
    x = np.linspace(A, B, N+1)
    h = x[1] - x[0]

    y = compose_cond_and_solve(lambd, x, N)
    nn = N * 10
    nh = B / nn

    values = [[sin(sqrt(lambd) *  (i * nh)), calc(y, i * nh, x, N)] for i in range(nn)]

    max_err = max(abs(th - real) for th, real in values)

    print(f"max_err={max_err}, h^2={h**2}")


if __name__ == "__main__":
    main()
