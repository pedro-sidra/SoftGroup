import torch
import time
from softgroup.util.fps import FPS
from scipy.stats import zscore
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import pandas as pd
import numpy as np


def normalize(arr):
    return arr / np.linalg.norm(arr)


def get_random_cube(
    cube_size_min=np.array([0.1, 0.1, 0.1]),
    cube_size_max=np.array([0.5, 0.5, 0.5]),
    cell_size=0.01,
    actual_cube=False,
    sphere=None,
):
    # random rotation in z
    rotation = R.from_euler("z", np.random.rand() * np.pi, degrees=False).as_matrix()

    # uniform random between size_min and size_max
    cube_size = cube_size_min + np.random.rand(3) * (cube_size_max - cube_size_min)

    if actual_cube:
        # Same dimensions on xyz
        points_x = np.arange(0, cube_size[0], cell_size)
        points_y = np.arange(0, cube_size[0], cell_size)
        points_z = np.arange(0, cube_size[0], cell_size)
    else:
        points_x = np.arange(0, cube_size[0], cell_size)
        points_y = np.arange(0, cube_size[1], cell_size)
        points_z = np.arange(0, cube_size[2], cell_size)

    x, y, z = np.meshgrid(points_x, points_y, points_z)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    z = z - z.mean()

    cube = np.stack([x, y, z])
    if sphere:
        cube = cube[:, (x**2 + y**2 + z**2) < cube_size.min() ** 2]

    cube = rotation @ cube

    return cube.T


def get_random_cube_random_point_reference(points, *args, **kwargs):
    point = np.random.randint(0, len(points))

    point = points[point]
    return get_random_cube(*args, **kwargs) + point.reshape((1, 3))


def get_random_cube_average_heighted_point_reference(points, *args, **kwargs):
    z_coordinate_zscores = zscore(points[2, :])

    # high absolute value of zscore -> low proba
    sample_probas = normalize(1 / (0.001 + np.abs(z_coordinate_zscores)))
    point = np.random.choice(np.arange(len(points)), p=sample_probas)

    point = points[point]
    return get_random_cube(*args, **kwargs) + point.reshape((1, 3))


def get_random_cubes_random_sampled_point_references(points, npoints=10, *args, **kwargs):
    idxs = np.random.randint(0, len(points), size=npoints)

    cubes = []

    for point in points[idxs]:
        cubes.append(get_random_cube(*args, **kwargs) + point.reshape((1, 3)))

    return np.vstack(cubes)

def get_random_cubes_FPS_sampled_point_references(points, npoints=10, *args, **kwargs):
    t0=time.time()
    fps= FPS(points, n_samples=npoints)
    fps.fit()
    points = fps.get_selected_pts()
    print(f"FPS took {time.time()-t0}")
    #points, indices 

    cubes = []

    for point in points:
        cubes.append(get_random_cube(*args, **kwargs) + point.reshape((1, 3)))

    return np.vstack(cubes)


if __name__ == "__main__":
    f = Path("./dataset/scannetv2/train/scene0000_00_inst_nostuff.pth")
    xyz, rgb, dummy_sem_label, dummy_inst_label = torch.load(f)

    pc = pd.DataFrame(
            dict(
                x=xyz[:,0],
                y=xyz[:,1],
                z=xyz[:,2],
                )
            )

    cubes = get_random_cubes_random_sampled_point_references(
        pc[["x", "y", "z"]].to_numpy(),
        npoints=500,
        cell_size=0.04,
        actual_cube=True,
        cube_size_min=np.array([0.1, 0.1, 0.1]),
        cube_size_max=np.array([0.5, 0.5, 0.5]),
        sphere=True,
    )
    cubes = pd.DataFrame(
        cubes,
        columns=["x", "y", "z"],
    )
    # cubes = []
    # for i in range(5):
    #     cubes.append(
    #         pd.DataFrame(
    #             get_random_cube_random_point_reference(
    #                 pc[["x", "y", "z"]].to_numpy(),
    #                 cell_size=0.01,
    #                 actual_cube=True,
    #                 cube_size_min=np.array([0.1, 0.1, 0.1]),
    #                 cube_size_max=np.array([1.0, 1.0, 1.0]),
    #             ),
    #             columns=["x", "y", "z"],
    #         )
    #     )
    pc = pd.concat([pc, cubes])

    pc.to_csv("output.txt")
