import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sbn  # Configuring Matplotlib
import matplotlib as mpl
from numba import jit
from tqdm import tqdm
from fastdtw import fastdtw
from dtaidistance import dtw

mpl.rcParams['figure.dpi'] = 300
savefig_options = dict(format="png", dpi=300, bbox_inches="tight")  # Computation packages

tmatrix_path = "tmatrix/"
gmatrix_path = "gmatrix/"

timage_path = "timages/"
gimage_path = "gimages/"

ignored_bpdy_parts_indice = [4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32, 33, 34,
                             35, 36, 37, 38, 39, 40, 41, 42, 43, 52]

def find_path(matrix:np.array):

    path = []
    path.append(tuple([0, 0]))

    i = 0
    j = 0

    while i != matrix.shape[0] - 1 or j != matrix.shape[1] - 1:

        if (i >= matrix.shape[0] - 1) and (j >= matrix.shape[1] - 1):
            return path

        elif i >= matrix.shape[0] - 1:
            i = matrix.shape[0] - 1
            path.append(tuple([i, j + 1]))
            j = j + 1

        elif j >= matrix.shape[1] - 1:
            j = matrix.shape[1] - 1
            path.append(tuple([i + 1, j]))
            i = i + 1
        else:

            up = matrix[i, j + 1]
            right = matrix[i + 1, j]
            diag = matrix[i + 1, j + 1]

            current_min = min(
                right,  # insertion
                up,  # deletion
                diag  # match
            )


            if current_min == diag:
                path.append(tuple([i + 1, j + 1]))
                i = i + 1
                j = j + 1
            elif current_min == up:
                path.append(tuple([i, j + 1]))
                j = j + 1
            else:
                path.append(tuple([i + 1, j]))
                i = i + 1

    return path



@jit(nopython=True)
def compute_accumulated_cost_matrix(C):
    """Compute the accumulated cost matrix given the cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n-1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m-1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D

@jit(nopython=True)
def compute_optimal_warping_path(D):
    """Compute the warping path given an accumulated cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        D (np.ndarray): Accumulated cost matrix

    Returns:
        P (np.ndarray): Optimal warping path
    """
    N = D.shape[0]
    M = D.shape[1]
    n = N - 1
    m = M - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        (n, m) = cell
    P.reverse()
    return np.array(P)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    flag = 0
    if flag == 0:
        matrix_path = tmatrix_path
    else:
        matrix_path = gmatrix_path

    for root, dirs, files in os.walk(matrix_path, topdown=True):

        if dirs:
            continue

        flags = [True if "." in dir else False for dir in dirs]

        folder = root.split("/")[-1]

        # print(dirs)
        # print(flags)

        if folder:

            if not os.path.exists(timage_path + "/" + folder):
                os.makedirs(timage_path + "/" + folder)

            for f in tqdm(files):
                # print(f)
                if f.endswith(".npy"):

                    data = np.load(root + "/" + f)
                    ndata = np.ones((data.shape[0], data.shape[1]))

                    for i in range(1, data.shape[0]):
                        for j in range(1, data.shape[1]):
                            score = 0
                            vector = data[i, j]

                            base = len(vector[:-1]) - len(ignored_bpdy_parts_indice)

                            for v in range(len(vector)):
                                if v in ignored_bpdy_parts_indice:
                                    continue
                                score += vector[v]

                            s = score / base

                            ndata[i, j] = s

                    ndata[0, 0] = 0

                    D =  compute_accumulated_cost_matrix(ndata)
                    P = compute_optimal_warping_path(D)

                    fig, ax = plt.subplots(figsize=(16, 12))
                    ax = sbn.heatmap(ndata, annot=False, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)
                    ax.invert_yaxis()
                    # Get the warp path in x and y directions
                    path_x = [p[0] for p in P]
                    path_y = [p[1] for p in P]

                    # Align the path from the center of each cell
                    path_xx = [x + 0.5 for x in path_x]
                    path_yy = [y + 0.5 for y in path_y]
                    ax.plot(path_xx, path_yy, color='red', linewidth=3, alpha=0.2)
                    if flag == 0:
                        fig.savefig(timage_path + "/" + folder + "/" + f.split(".")[0] + ".png", **savefig_options)
                        np.save(timage_path + "/" + folder + "/" + f.split(".")[0] + ".npy", ndata)
                        np.save(timage_path + "/" + folder + "/" + f.split(".")[0] + "_path.npy", P)

                    else:
                        fig.savefig(gimage_path + "/" + folder + "/" + f.split(".")[0]  + ".png", **savefig_options)
                        np.save(timage_path + "/" + folder + "/" + f.split(".")[0] + ".npy", ndata)
                        np.save(timage_path + "/" + folder + "/" + f.split(".")[0] + "_path.npy", P)

                    del data