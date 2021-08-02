import numpy as np
import os
import math
from tqdm import tqdm
import multiprocessing as mp

print("Number of processors: ", mp.cpu_count())

video_path = "./npy/"
matrix_path = "./"

motion_dict = {}


# ignored_bpdy_parts_indice = [4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32, 33, 34,35, 36, 37, 38, 39, 40, 41, 42, 43]


def initial_vector_matrix(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n + 1, m + 1, 52 + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.zeros(52 + 1)
    dtw_matrix[0, 0] = np.zeros(52 + 1)

    return dtw_matrix


def get_matrix(name: str, compared_name: str):
    if compared_name == name:
        return

    translation = motion_dict[name][:, :, 0, :]
    quaternion = motion_dict[name][:, :, 1, :]

    cn_translation = motion_dict[compared_name][:, :, 0, :]
    cn_quaternion = motion_dict[compared_name][:, :, 1, :]

    # initial matrix
    zero_matrix = initial_vector_matrix(quaternion, cn_quaternion)
    print()
    print(name + "===" + compared_name)
    print(zero_matrix.shape[0])
    print(zero_matrix.shape[1])

    # set distance in zero matrix
    for frame in range(quaternion.shape[0]):

        # accumulated_delta_hip = 0

        for cn_frame in range(cn_quaternion.shape[0]):
            # hip quaternion delta

            vector = []

            for b in range(52):
                current_frame_b_quaternion = quaternion[frame, b, 1]
                current_frame_cn_b_quaternion = cn_quaternion[cn_frame, b, 1]
                delta_b_quaternion = 1.0 - math.pow(
                    np.dot(current_frame_b_quaternion, current_frame_cn_b_quaternion), 2)
                vector.append(delta_b_quaternion)

            # hip translation delta
            current_frame_hip_translation = translation[frame, 0, 0]
            current_frame_cn_hip_translation = cn_translation[cn_frame, 0, 0]
            delta_hip_translation = np.linalg.norm(
                current_frame_hip_translation - current_frame_cn_hip_translation).item()
            vector.append(delta_hip_translation)

            # plot distance
            zero_matrix[frame + 1, cn_frame + 1] = np.array(vector)

    np.save(matrix_path + "/" + name + "/" + compared_name + ".npy", zero_matrix)
    del zero_matrix

    return name, compared_name


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for root, dirs, files in os.walk(video_path, topdown=True):

        for f in files:
            # print(f)
            if f.endswith(".npy"):
                data = np.load(video_path + "/" + f)
                f_we = f.split(".")[0]

                motion_dict[f_we] = data

    # pool = mp.Pool(mp.cpu_count())
    #
    # pair_list = pool.starmap_async(get_palette, ).get()
    #
    # pool.close()

    for n in tqdm(motion_dict):

        # create folder
        if not os.path.exists(matrix_path + "/" + n):
            os.makedirs(matrix_path + "/" + n)

        pool = mp.Pool(mp.cpu_count())

        pair_list = pool.starmap_async(get_matrix, [(n, cn) for cn in motion_dict]).get()

        pool.close()
