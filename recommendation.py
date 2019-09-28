import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_data_rating():
    start = time.time()
    Data_rate_file = open("ratings.dat", "r")
    Line_read_file = Data_rate_file.readlines()
    A = []

    for i in range(len(Line_read_file)):
        a = Line_read_file[i].split("::")
        a = a[0:3]
        a = map(float, a)
        A.append(a)

    df = pd.DataFrame(A, columns=['User', 'Movie', 'Ratings'])

    y = df.pivot(index='User', columns='Movie', values='Ratings')
    y = y.fillna(0)

    Final = y.as_matrix(columns=None)

    end = time.time() - start

    return Final


def check_rate_zeros(n):
    listofzeros = [0] * n
    return listofzeros


def plotter(Y, K):
    n_users = Y.shape[0]
    delta = []

    for i in range(K):
        Y_dash = Y[:, 0:i + 1].max(axis=1)
        Sum_Y = Y_dash.sum() / n_users

        delta.append(Sum_Y)

    return delta


def hamming_simi(X, A):
    X_dash = X.max(axis=0)
    Sum_X_dash = X_dash.sum() / len(X)

    a = np.array(A).max(axis=0)
    Sum_a = a.sum() / len(X)

    v = Sum_X_dash - Sum_a

    return v


def Greedy_file(X, K):
    n_users = X.shape[0]

    A = check_rate_zeros(n_users)
    b_val = []

    TIME = []

    value = []

    Time_conter = time.time()

    for j in range(X.shape[1]):
        a_dash = X[:, j]
        # a_dash = a_dash.max(axis = 1)
        Sum_a_dash = a_dash.sum() / n_users

        a = np.array(A).max(axis=0)
        Sum_a = a.sum() / n_users

        v = Sum_a_dash - Sum_a

        value.append(v)

    v_arg = np.argsort(value)

    v_arg = v_arg[::-1]

    b = v_arg[0]
    b_val.append(b)
    A = (X[:, b])

    e_t = time.time() - Time_conter
    TIME.append(e_t)

    for k in range(K - 1):

        Time_conter = time.time()

        d = hamming_simi((X[:, v_arg[k + 1]]), (A))
        if (value[k + 2] < d):
            b = v_arg[k + 1]
            b_val.append(b)

            A = np.c_[A, (X[:, b])]

        elif (value[k + 2] > d):
            value[k + 1] = d
            v_arg = np.argsort(value)

            v_arg = v_arg[::-1]
            b = v_arg[k + 1]
            b_val.append(b)

            A = np.c_[A, (X[:, b])]

        e_t = time.time() - Time_conter
        TIME.append(e_t)

    return A, v_arg, TIME


X = load_data_rating()

K = 20
PP = [5,10,25,50]
EE = []
for K in PP:
    movies_list1, Rank1, T1 = Greedy_file(X, K)

    delta1 = plotter(movies_list1, K)
    T1 = list(np.cumsum(T1))
    l = []

    for i in range(K):
        l.append(i)
    EE.append(max(delta1))

plt.grid()

print(l)

plt.xlabel('# recommendations')
plt.ylabel('DCG')
plt.title('DCG')
red_patch = mpatches.Patch(color='red', label='Greedy Algorithm')
plt.legend(handles=[red_patch])

plt.plot(PP, EE)
plt.show()

