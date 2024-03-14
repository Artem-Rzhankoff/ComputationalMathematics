import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

#read data from result1.txt
data_fe = []
with open("result1.txt", "r") as file:
    for line in file.readlines():
        f_list = [float(i) for i in line.split(" ") if i.strip()]
        data_fe.append([int(f_list[0]), float(f_list[1]), float(f_list[2])])


df = pd.DataFrame(data_fe, columns=['Grid size', 'Time (1 thread)', 'Time (12 threads)'])

table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

table.set_fontsize(14)
table.scale(1,4)
ax.axis('off')

plt.savefig("results/test1.png", bbox_inches="tight")
#plt.show()

#read data from result2.txt
for i in 2, 3, 4:
    data_fe = []
    with open("result{num:d}.txt".format(num=i), "r") as file:
        for line in file.readlines():
            f_list = [float(i) for i in line.split(" ") if i.strip()]
            data_fe.append([int(f_list[0]), float(f_list[1]), float(f_list[2]), float(f_list[3])])
    df = pd.DataFrame(data_fe, columns=['Grid size', 'Time (1 thread)', 'Time (12 threads)', 'Approximation error'])

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    table.set_fontsize(14)
    table.scale(1,4)
    ax.axis('off')
    plt.savefig("results/test{num:d}.png".format(num=i), bbox_inches="tight")

plt.show()
#evaluate the effectiveness of the parallel version for various functions

data = [[], [], []]
for i in 2, 3, 4:
    with open("result{num:d}.txt".format(num=i), "r") as file:
        for line in file.readlines():
            f_list = [float(i) for i in line.split(" ") if i.strip()]
            # add grid size and efficiency increase coeff
            coeff = float(f_list[1]) / float(f_list[2])
            data[i-2].append([int(f_list[0]), coeff])

plt.axis([0, 2000, 0, 5])
plt.title('Эффективность параллельной версии', fontsize=20, fontname='Times New Roman')
plt.xlabel('Размер сетки', color='gray')
plt.ylabel('Отношение скорости вычислений', color='gray')

grid_size_num = 6

for i in 2, 3, 4:
    xs = [data[i-2][j][0] for j in range(grid_size_num)]
    ys = [data[i-2][j][1] for j in range(grid_size_num)]
    plt.plot(xs, ys, label='Функция №{num:d}'.format(num=i))

plt.legend()
plt.savefig("results/compare_efficiency.png")

plt.show()
#evaluate the approximation error coefficient for various function (with various grids)

data = [[], [], []]
for i in 2, 3, 4:
    with open("result{num:d}.txt".format(num=i), "r") as file:
        for line in file.readlines():
            f_list = [float(i) for i in line.split(" ") if i.strip()]
            # add grid size and efficiency increase coeff
            data[i-2].append([int(f_list[0]), float(f_list[3])])

plt.axis([0, 2000, 0, 2.5])
plt.title('Точность вычислений', fontsize=20, fontname='Times New Roman')
plt.xlabel('Размер сетки', color='gray')
plt.ylabel('Средняя относительная ошибка', color='gray')

for i in 2, 3, 4:
    xs = [data[i-2][j][0] for j in range(grid_size_num)]
    ys = [data[i-2][j][1] for j in range(grid_size_num)]
    plt.plot(xs, ys, label='Функция №{num:d}'.format(num=i))

plt.legend()
plt.savefig("results/approximation_error.png")
