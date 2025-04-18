import os
import numpy as np
import re

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import (
    register_visual_change_object,
    register_object,
)


class BoubaKikiObject(MujocoXMLObject):
    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        # make sure custom path is an absolute path
        custom_path = os.path.join(
            str(absolute_path),
            f"assets/bouba_kiki/{obj_name}/{obj_name}.xml",
        ),
        # make sure the custom path is also an xml file
        super().__init__(
            custom_path[0],
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}


@register_object
class Bouba(BoubaKikiObject):
    def __init__(self,
                 name="bouba",
                 obj_name="bouba",
                 ):
        super().__init__(
            name=name,
            obj_name=obj_name,
        )

        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None


@register_object
class Kiki(BoubaKikiObject):
    def __init__(self,
                 name="kiki",
                 obj_name="kiki",
                 ):
        super().__init__(
            name=name,
            obj_name=obj_name,
        )

        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None
