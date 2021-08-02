import matplotlib.pyplot
import os
import numpy as np

motion_dict = {}
video_path = "./npy/"

body = [5, 4, 3, 2, 0]
lshoulder = [4, 25, 3]
rshoulder = [4, 6, 3]

lefta = [25, 26, 27, 28]
refta = [6, 7, 8, 9]

lleg = [0, 44, 45, 46, 47]
rleg = [0, 48, 49, 50, 51]

if __name__ == '__main__':

    for root, dirs, files in os.walk(video_path, topdown=True):

        for f in files:
            # print(f)
            if f.endswith(".npy"):
                data = np.load(video_path + "/" + f)
                f_we = f.split(".")[0]

                motion_dict[f_we] = data

    for name in motion_dict:
        data = motion_dict[name]
        print(name)
        for frame in range(data.shape[0]):

            xs = data[frame, :, 0, 0]
            ys = data[frame, :, 0, 1]
            zs = data[frame, :, 0, 2]

            body_xs = [xs[i] for i in body]
            body_ys = [ys[i] for i in body]
            body_zs = [zs[i] for i in body]

            lshoulder_xs = [xs[i] for i in lshoulder]
            lshoulder_ys = [ys[i] for i in lshoulder]
            lshoulder_zs = [zs[i] for i in lshoulder]

            rshoulder_xs = [xs[i] for i in rshoulder]
            rshoulder_ys = [ys[i] for i in rshoulder]
            rshoulder_zs = [zs[i] for i in rshoulder]

            lefta_xs = [xs[i] for i in lefta]
            lefta_ys = [ys[i] for i in lefta]
            lefta_zs = [zs[i] for i in lefta]

            refta_xs = [xs[i] for i in refta]
            refta_ys = [ys[i] for i in refta]
            refta_zs = [zs[i] for i in refta]

            lleg_xs = [xs[i] for i in lleg]
            lleg_ys = [ys[i] for i in lleg]
            lleg_zs = [zs[i] for i in lleg]

            rleg_xs = [xs[i] for i in rleg]
            rleg_ys = [ys[i] for i in rleg]
            rleg_zs = [zs[i] for i in rleg]

            fig = matplotlib.pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')
            #
            # ax.plot3D(body_xs, body_ys, body_zs, color='b')
            # ax.plot3D(lshoulder_xs, lshoulder_ys, lshoulder_zs, color='r')
            # ax.plot3D(rshoulder_xs, rshoulder_ys, rshoulder_zs, color='g')
            # ax.plot3D(lefta_xs, lefta_ys, lefta_zs, color='k')
            # ax.plot3D(refta_xs, refta_ys, refta_zs, color='y')
            # ax.plot3D(lleg_xs, lleg_ys, lleg_zs, color='c')
            # ax.plot3D(rleg_xs, rleg_ys, rleg_zs, color='m')

            ax.plot3D(body_xs, body_zs, body_ys, color='b')
            # ax.plot3D(lshoulder_xs, lshoulder_zs, lshoulder_ys, color='r')
            # ax.plot3D(rshoulder_xs, rshoulder_zs, rshoulder_ys, color='g')
            ax.plot3D(lefta_xs, lefta_zs, lefta_ys, color='k')
            ax.plot3D(refta_xs, refta_zs, refta_ys, color='y')
            ax.plot3D(lleg_xs, lleg_zs, lleg_ys, color='c')
            ax.plot3D(rleg_xs, rleg_zs, rleg_ys, color='m')

            matplotlib.pyplot.show()

        break