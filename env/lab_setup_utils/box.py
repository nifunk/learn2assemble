from typing import Sequence, Optional, Dict, Any

import numpy as np
import pybullet

from assembly_gym.util import Transformation


class Box:
    def __init__(self, extents: Sequence[float], rgba_color: Sequence[float] = (0.5, 0.5, 0.5, 1.0),
                 mass: Optional[float] = None, pose: Optional[Transformation] = None,
                 create_visual_shape: bool = True, create_collision_shape: bool = True):
        assert len(extents) == 3
        assert len(rgba_color) == 4
        self.__extents = np.array(extents)
        self.__rgba_color = rgba_color
        # super(Box, self).__init__(mass, pose, create_visual_shape, create_collision_shape)


    def get_collision_shape_kwargs(self) -> Dict[str, Any]:
        return {
            "shapeType": pybullet.GEOM_BOX,
            "halfExtents": self.__extents / 2
        }

    def get_visual_shape_kwargs(self) -> Dict[str, Any]:
        return {
            "shapeType": pybullet.GEOM_BOX,
            "halfExtents": self.__extents / 2,
            "rgbaColor": self.__rgba_color
        }
