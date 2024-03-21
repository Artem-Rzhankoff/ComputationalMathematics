from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

#read data from result1.i.txt

#по 5 замеров для каждого размера сетки
data = [[], [], [], [], []]

for i in range(0, 5):
    with open("result1.{run:d}.txt".format(run=i+1), "r") as file:
        for line in file.readlines():
            f_list = [float(i) for i in line.split(" ") if i.strip()]
            data[i].append([int(f_list[0]), float(f_list[1]), float(f_list[2])])

data_fe = []
#теперь проходим по грид сайзам
for i in range(0, 6):
    t_single = []
    t_multi = []
    for j in range(0, 5):
        t_single.append(data[j][i][1])
        t_multi.append(data[j][i][2])

    average_single = np.mean(t_single)
    average_multi = np.mean(t_multi)
    
    a = stats.t.interval(0.95, df=len(t_single)-1, loc=np.mean(t_single), scale=stats.sem(t_single))
    b = stats.t.interval(0.95, df=len(t_multi)-1, loc=np.mean(t_multi), scale=stats.sem(t_multi))

    data_fe.append([data[0][i][0], round(average_single, 1), pd.Interval(round(a[0], 1), round(a[1], 1)), round(average_multi, 1),
                     pd.Interval(round(b[0], 1), round(b[1], 1))])


df = pd.DataFrame(data_fe, columns=['Grid size', '1 thread\n(average)', '1 thread\n(conf interval)', '12 threads\n(average)', '12 threads\n(conf interval)'])

table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

table.set_fontsize(10)
table.scale(1,4)
ax.axis('off')

plt.savefig("results/test1.png", bbox_inches="tight")

#read date from result2.i.txt and the rest

data = [[], [], [], []]

for i in range(1, 4):
    # по экспериментам
    data[i-1] = [[], [], [], [], []]
    for j in range(0, 5):
        with open("result{num:d}.{run:d}.txt".format(run=j+1, num=i+1), "r") as file:
            for line in file.readlines():
                f_list = [float(i) for i in line.split(" ") if i.strip()]
                data[i-1][j].append([int(f_list[0]), float(f_list[1]), float(f_list[2]), float(f_list[3])])


for k in range(1, 4):
    f = open("result{num:d}.txt".format(num=k+1), "w")
    data_fe = []
    #теперь проходим по грид сайзам
    for i in range(0, 6):
        t_single = []
        t_multi = []
        for j in range(0, 5):
            t_single.append(data[k-1][j][i][1])
            t_multi.append(data[k-1][j][i][2])

        average_single = np.mean(t_single)
        average_multi = np.mean(t_multi)
        accuracy = 3 if k == 2 else 2
        f.write("{x:d} {y:f} {z:f} {h:f}\n".format(x=data[k-1][0][i][0], y=round(average_single, accuracy), z=round(average_multi, accuracy), h=data[k-1][0][i][3]))
        
        a = stats.t.interval(0.95, df=len(t_single)-1, loc=np.mean(t_single), scale=stats.sem(t_single))
        b = stats.t.interval(0.95, df=len(t_multi)-1, loc=np.mean(t_multi), scale=stats.sem(t_multi))

        data_fe.append([data[k-1][0][i][0], round(average_single, accuracy), pd.Interval(round(a[0], accuracy), round(a[1], accuracy)), round(average_multi, accuracy),
                        pd.Interval(round(b[0], accuracy), round(b[1], accuracy))])
        
        df = pd.DataFrame(data_fe, columns=['Grid size', '1 thread\n(average)', '1 thread\n(conf interval)', '12 threads\n(average)', '12 threads\n(conf interval)'])

        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

        table.set_fontsize(10)
        table.scale(1,4)
        ax.axis('off')

        plt.savefig("results/test{num:d}.png".format(num=k+1), bbox_inches="tight")
    f.close()

plt.show()

#evaluate the effectiveness of the parallel version for various functions

data = [[], [], []]
for i in range(2, 5):
    with open("result{num:d}.txt".format(num=i), "r") as file:
        for line in file.readlines():
            print(i)
            f_list = [float(i) for i in line.split(" ") if i.strip()]
            # add grid size and efficiency increase coeff
            coeff = round(float(f_list[1]) / float(f_list[2]), 3)
            data[i-2].append([int(f_list[0]), coeff])

plt.axis([0, 1500, 0, 5])
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

plt.axis([0, 1500, 0, 5.5])
plt.title('Точность вычислений', fontsize=20, fontname='Times New Roman')
plt.xlabel('Размер сетки', color='gray')
plt.ylabel('Средняя относительная ошибка', color='gray')

for i in 2, 3, 4:
    xs = [data[i-2][j][0] for j in range(grid_size_num)]
    ys = [data[i-2][j][1] for j in range(grid_size_num)]
    plt.plot(xs, ys, label='Функция №{num:d}'.format(num=i))

plt.legend()
plt.savefig("results/approximation_error.png")
