import numpy as np
from matplotlib import pyplot as plt

from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import (
    register_task_info,
    get_task_info,
    generate_bddl_from_task_info,
)

@register_mu(scene_type="living_room")
class BoubakikiScene1(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "living_room_table": 1,
        }

        object_num_info = {
            "kiki": 1,
            "bouba": 1,
            "basket": 1,
            "tomato_sauce": 1,
        }

        super().__init__(
            workspace_name="living_room_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.26],
                region_name="basket_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, -0.10],
                region_name="bouba_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.06],
                region_name="kiki_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, -0.20],
                region_name="tomato_sauce_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "bouba_1", "living_room_table_bouba_init_region"),
            ("On", "kiki_1", "living_room_table_kiki_init_region"),
            ("On", "basket_1", "living_room_table_basket_init_region"),
            ("On", "tomato_sauce_1", "living_room_table_tomato_sauce_init_region"),
        ]
        return states
def main():
    scene_name = "Boubakiki_Scene1"
    language = "pick up the round item and place it in the basket"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["bouba_1", "kiki_1", "tomato_sauce_1"],
        goal_states=[
            (""),
            ("In", "bouba_1", "basket_1_contain_region"),
        ],
    )

    scene_name = "Boubakiki_Scene1"
    language = "pick up bouba and place it in the basket"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["bouba_1", "kiki_1", "tomato_sauce_1"],
        goal_states=[
            (""),
            ("In", "bouba_1", "basket_1_contain_region"),
        ],
    )

    scene_name = "Boubakiki_Scene1"
    language = "pick up the spike item and place it in the basket"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["bouba_1", "kiki_1", "tomato_sauce_1"],
        goal_states=[
            (""),
            ("In", "kiki_1", "basket_1_contain_region"),
        ],
    )

    scene_name = "Boubakiki_Scene1"
    language = "pick up kiki and place it in the basket"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["bouba_1", "kiki_1", "tomato_sauce_1"],
        goal_states=[
            (""),
            ("In", "kiki_1", "basket_1_contain_region"),
        ],
    )

    scene_name = "Boubakiki_Scene1"
    language = "pick up the cube and place it in the basket"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["kiki_1", "bouba_1", "tomato_sauce_1"],
        goal_states=[
            (""),
            ("In", "kiki_1", "basket_1_contain_region"),
        ],
    )

    scene_name = "Boubakiki_Scene1"
    language = "pick up the tomato sauce and place it in the basket"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["kiki_1", "bouba_1", "tomato_sauce_1"],
        goal_states=[
            ("In", "tomato_sauce_1", "basket_1_contain_region"),
        ],
    )

    bddl_file_names, failures = generate_bddl_from_task_info("../libero/libero/bddl_files/libero_object")
    print(bddl_file_names)

    with open(bddl_file_names[0], "r") as f:
        print(f.read())

    from libero.libero.envs import OffScreenRenderEnv
    from IPython.display import display
    from PIL import Image

    import torch
    import torchvision

    env_args = {
        "bddl_file_name": bddl_file_names[0],
        "camera_heights": 256,
        "camera_widths": 256
    }

    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    # display(Image.fromarray(obs["agentview_image"][::-1]))

    img = obs["agentview_image"]

    # Optionally flip vertically if needed
    # img = img[::-1]
    plt.imshow(img)
    plt.axis("off")
    plt.title("Initial Observation")
    plt.show()

if __name__ == "__main__":
    main()
