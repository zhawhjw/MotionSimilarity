import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sbn  # Configuring Matplotlib
import matplotlib as mpl
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 300
savefig_options = dict(format="png", dpi=300, bbox_inches="tight")  # Computation packages

matrix_path = "./matrix/"

ignored_bpdy_parts_indice = [4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32, 33, 34,
                             35, 36, 37, 38, 39, 40, 41, 42, 43, 52]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    for root, dirs, files in os.walk(matrix_path, topdown=True):

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

                # if not os.path.exists(f.split(".")[0]):
                #     os.makedirs(f.split(".")[0] + "/" + n)

                fig, ax = plt.subplots(figsize=(16, 12))
                ax = sbn.heatmap(ndata, annot=False, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)
                ax.invert_yaxis()

                fig.savefig(f + ".png", **savefig_options)

                del data

