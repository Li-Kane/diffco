import sapien
import mplib
import numpy as np
import trimesh
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.envs.sapien_env import BaseEnv
from mplib.pymp import Pose
from .robot_interface_base import RobotInterfaceBase
from gymnasium.core import Env

class ManiskillEnv(RobotInterfaceBase):
    def __init__(
            self,
            name='',
            device='cpu',
            env: Env=None,
            urdf_path=None,
            srdf_path=None,
            move_group=None,
        ):
        super().__init__(name, device)
        if env is None:
            raise ValueError('env cannot be None')
        else:
            base_env: BaseEnv = env.unwrapped
            self.robot = base_env.agent.robot

        if urdf_path is None and srdf_path is None:
            urdf_path = base_env.agent.urdf_path
            srdf_path = base_env.agent.urdf_path.replace('.urdf', '.srdf')

        self.planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=[link.get_name() for link in self.robot.get_links()],
            user_joint_names=[joint.get_name() for joint in self.robot.get_active_joints()],
            move_group=move_group,
        )
        sapien_pose : sapien.Pose = to_sapien_pose(self.robot.pose)
        base_pose = Pose(sapien_pose.get_p(), sapien_pose.get_q())
        self.planner.set_base_pose(base_pose)

        # setup environment point clouds
        collision_pts = np.ndarray((0, 3))
        for actor in base_env.scene.actors.values():
            print(actor.name)
            for mesh in actor.get_collision_meshes(to_world_frame=True):
                pts, _ = trimesh.sample.sample_surface(mesh, int(mesh.area * 1000))
                collision_pts = np.vstack((collision_pts, pts))
        for articulation in base_env.scene.articulations.values():
            # don't add the robot to the planning environment
            if(articulation.get_name() != self.robot.get_name()):
                for mesh in articulation.get_collision_meshes(to_world_frame=True):
                    pts, _ = trimesh.sample.sample_surface(mesh, int(mesh.area * 1000))
                    collision_pts = np.vstack((collision_pts, pts))
        # filter out points too close to the ground
        collision_pts = [point for point in collision_pts if point[2] > 0.2]
        self.planner.update_point_cloud(collision_pts)

        # get robot joint limits
        self.joint_limits = self.planner.joint_limits

    # return num_configs amount of random configurations
    # which is an array of shape (num_configs, num_dofs)
    def rand_configs(self, num_configs):
        return np.random.rand(num_configs, len(self.joint_limits)) * (self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0]

    # for q = [batch_size x n_dofs], return a list of [batch_size] bools
    # indicating whether each configuration is in collision
    def collision(self, q):
        return [(len(self.planner.check_for_self_collision(state=qpos)) > 0 or len(self.planner.check_for_env_collision(state=qpos)) > 0) for qpos in q]

    # for q = [batch_size x n_dofs], return a dictionary of
    # {link_name: [translation, rotation]} for each link
    # where translation is of shape (batch_size, 3,) and rotation is of shape (batch_size, 4,)
    def compute_forward_kinematics_all_links(self, q, return_collision=False):
        # initialize our dictionary
        link_poses = {}
        model = self.planner.pinocchio_model
        for link_name in self.planner.link_name_2_idx:
            link_poses[link_name] = [np.empty((0, 3)), np.empty((0, 4))]
        for qpos in q:
            model.compute_forward_kinematics(qpos)
            for link_name in self.planner.link_name_2_idx:
                pose = model.get_link_pose(self.planner.link_name_2_idx[link_name])
                link_poses[link_name][0] = np.vstack((link_poses[link_name][0], pose.p))
                link_poses[link_name][1] = np.vstack((link_poses[link_name][1], pose.q))
        return link_poses

# return collisions?
# rotation is a quaternion of 3x3 matrix?