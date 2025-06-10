import time
import sapien
import mplib
import numpy as np
import trimesh
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.envs.sapien_env import BaseEnv
from mplib.pymp import Pose
from .robot_interface_base import RobotInterfaceBase
from gymnasium.core import Env
import pytorch_kinematics as pk
import torch

class ManiskillRobot(RobotInterfaceBase):
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
            user_link_names=[link.name for link in self.robot.links],
            user_joint_names=[joint.name for joint in self.robot.active_joints],
            move_group=move_group,
        )
        sapien_pose : sapien.Pose = to_sapien_pose(self.robot.pose)
        base_pose = Pose(sapien_pose.get_p(), sapien_pose.get_q())
        self.planner.set_base_pose(base_pose)

        # setup environment point clouds
        collision_pts = np.ndarray((0, 3))
        for actor in base_env.scene.actors.values():
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
        self.limits = torch.tensor(self.planner.joint_limits, dtype=torch.float64)
        self.dof = len(self.joint_limits)

        # pytorch kinematics
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read())

    # return num_configs amount of random configurations
    # which is an array of shape (num_configs, num_dofs)
    def rand_configs(self, num_configs):
        rand = torch.rand(num_configs, self.dof, device=self._device, dtype=torch.float64)
        return torch.tensor(rand * (self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0], dtype=torch.float64)

    # for q = [batch_size x n_dofs], return a list of [batch_size] bools
    # indicating whether each configuration is in collision
    def collision(self, q, other=None):
        return torch.tensor([
            (len(self.planner.check_for_self_collision(state=qpos)) > 0 or 
                len(self.planner.check_for_env_collision(state=qpos)) > 0)
            for qpos in q
        ], dtype=torch.bool, device=self._device)

    # for q = [batch_size x n_dofs], return a dict of
    # {link_name: [translation, rotation]} for each link
    # where translation is of shape (batch_size, 3,) and rotation is of shape (batch_size, 9,)
    def compute_forward_kinematics_all_links(self, q, return_collision=False):
        if return_collision:
            raise NotImplementedError('Collision checking not implemented for forward kinematics')
        batch_size = q.shape[0]
        # initialize our dictionary
        link_poses = {}
        for link_name in self.planner.link_name_2_idx:
            link_poses[link_name] = (torch.empty(batch_size, 3), torch.empty(batch_size, 9))
        model = self.planner.pinocchio_model
        for i in range(batch_size):
            model.compute_forward_kinematics(q[i])
            for link_name in self.planner.link_name_2_idx:
                pose = model.get_link_pose(self.planner.link_name_2_idx[link_name])
                link_poses[link_name][0][i] = torch.tensor(pose.p, dtype=torch.float64)

                trans_matrix = pose.to_transformation_matrix()
                rotation = torch.tensor(trans_matrix[:3,:3])
                link_poses[link_name][1][i] = torch.flatten(rotation)
        return link_poses

    # compute forward kinematics for the robot 
    # input : q = (batch_size, n_dofs) tensor
    # output = (batch_size, 3, num_links) tensor
    def fkine(self, q, return_collision=False, reuse=False):
        if return_collision:
            raise NotImplementedError('Collision checking not implemented for forward kinematics')
        if reuse:
            raise NotImplementedError('Reuse not implemented for forward kinematics')
        q = q.to(torch.float32)
        batch_size = q.shape[0]
        links = {}
        link_names = self.chain.get_link_names()
        for link_name in link_names:
            links[link_name] = torch.empty(batch_size, 3)
        for i in range(batch_size):
            fk_dict = self.chain.forward_kinematics(q[i])
            for link_name in fk_dict.keys():
                # link_idx = self.chain.frame_to_idx[link_name]
                links[link_name][i] = fk_dict[link_name]._matrix[:, :3, 3]
        # (29, batch_size, 3)
        fk_tensor_list = []
        for link_name in link_names:
            fk_tensor_list.append(links[link_name])
        fk_tensor = torch.stack(fk_tensor_list, dim=-1)
        # (29, batch_size, 3) -> (batch_size, 29, 3)
        return fk_tensor.to(torch.float64)