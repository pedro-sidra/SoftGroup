import numpy as np
import numpy as np
from softgroup.data.scannetv2 import ScanNetDataset

from softgroup.util.generate_occlusions import (
    get_random_cubes_random_sampled_point_references,
    get_random_cubes_FPS_sampled_point_references,
)


class SculptingPreTraining(ScanNetDataset):
    def add_random_cubes(self, xyz, rgb, semantic_label, instance_label):
        cubes = get_random_cubes_random_sampled_point_references(
            xyz,
            npoints=500,
            cell_size=0.04,
            actual_cube=True,
            cube_size_min=np.array([0.1, 0.1, 0.1]),
            cube_size_max=np.array([0.5, 0.5, 0.5]),
            sphere=True,
        )

        xyz = np.vstack([xyz, cubes])

        rand_colors = 2*np.random.rand(*cubes.shape)-1
        rgb = np.vstack([rgb, rand_colors])

        semantic_label = np.hstack(
            [np.ones_like(semantic_label), np.zeros(cubes.shape[0])]
        )

        instance_label = np.hstack([-1 * np.ones(instance_label.shape[0]), -1 * np.ones(cubes.shape[0])])

        return xyz, rgb, semantic_label, instance_label

    def transform_train(
        self,
        xyz,
        rgb,
        semantic_label,
        instance_label,
    ):

        return super().transform_train(
            *self.add_random_cubes(
                xyz,
                rgb,
                semantic_label,
                instance_label,
            )
        )

    def transform_test(self, xyz, rgb, semantic_label, instance_label):

        return super().transform_test(
            *self.add_random_cubes(
                xyz,
                rgb,
                semantic_label,
                instance_label,
            )
        )
