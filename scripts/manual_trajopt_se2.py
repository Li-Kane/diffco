import sys
import json
from diffco import DiffCo, MultiDiffCo
from diffco import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
from diffco.model import RevolutePlanarRobot
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects
from diffco import utils
from diffco.Obstacles import FCLObstacle
from time import time
from scipy.optimize import minimize

def original_traj_optimize(robot, dist_est, start_cfg, target_cfg, history=False):
    # There is a slightly different version in speed_compare.py,
    # which allows using SLSQP instead of Adam, allows
    # inputting an initial solution other than straight line,
    # and is better modularly written.
    # That one with SLSQP is more recommended to use.
    N_WAYPOINTS = 20
    NUM_RE_TRIALS = 10
    UPDATE_STEPS = 200
    dif_weight = 1
    max_move_weight = 10
    collision_weight = 10
    safety_margin = torch.FloatTensor([-12, -1.2])#([-8.0, -0.8]) #
    lr = 5e-1
    seed = 1234
    torch.manual_seed(seed)

    lowest_cost_solution = None
    lowest_cost = np.inf
    lowest_cost_trial = None
    lowest_cost_step = None
    best_valid_solution = None
    best_valid_cost = np.inf
    best_valid_step = None
    best_valid_trial = None
    
    trial_histories = []

    found = False
    for trial_time in range(NUM_RE_TRIALS):
        path_history = []
        if trial_time == 0:
            init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS))
        else:
            init_path = (torch.rand(N_WAYPOINTS, robot.dof))*np.pi*2-np.pi
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)

        for step in range(UPDATE_STEPS):
            opt.zero_grad()
            collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
            control_points = robot.fkine(p)
            max_move_cost = torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-0.3**2, min=0).sum()
            diff = (control_points[1:]-control_points[:-1]).pow(2).sum()
            constraint_loss = collision_weight * collision_score + max_move_weight * max_move_cost
            objective_loss = dif_weight * diff
            loss = objective_loss + constraint_loss
            loss.backward()
            p.grad[[0, -1]] = 0.0
            opt.step()
            p.data[:, 2] = utils.wrap2pi(p.data[:, 2])
            if history:
                path_history.append(p.data.clone())
            if loss.data.numpy() < lowest_cost:
                lowest_cost = loss.data.numpy()
                lowest_cost_solution = p.data.clone()
                lowest_cost_step = step
                lowest_cost_trial = trial_time
            if constraint_loss <= 1e-2:
                if objective_loss.data.numpy() < best_valid_cost:
                    best_valid_cost = objective_loss.data.numpy()
                    best_valid_solution = p.data.clone()
                    best_valid_step = step
                    best_valid_trial = trial_time
            if constraint_loss <= 1e-2 or step % (UPDATE_STEPS/5) == 0 or step == UPDATE_STEPS-1:
                print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
                    trial_time, step, 
                    collision_score.item(), collision_weight,
                    max_move_cost.item(), max_move_weight,
                    diff.item(), dif_weight,
                    loss.item()))
        trial_histories.append(path_history)
        
        if best_valid_solution is not None:
            found = True
            break
    if not found:
        print('Did not find a valid solution after {} trials!\
            Giving the lowest cost solution'.format(NUM_RE_TRIALS))
        solution = lowest_cost_solution
        solution_step = lowest_cost_step
        solution_trial = lowest_cost_trial
    else:
        solution = best_valid_solution
        solution_step = best_valid_step
        solution_trial = best_valid_trial
    path_history = trial_histories[solution_trial] # Could be empty when history = false
    if not path_history:
        path_history.append(solution)
    else:
        path_history = path_history[:(solution_step+1)]
    return solution, path_history, solution_trial, solution_step

def adam_traj_optimize(robot, dist_est, start_cfg, target_cfg, options):
    N_WAYPOINTS = options['N_WAYPOINTS'] # 20
    NUM_RE_TRIALS = options['NUM_RE_TRIALS'] # 10
    MAXITER = options['MAXITER'] # 200
    history = options['history']

    dif_weight = 1 # This should NOT be changed
    max_move_weight = 10
    collision_weight = 10
    joint_limit_weight = 10
    safety_margin = options['safety_margin']
    lr = 5e-1
    seed = options['seed']
    torch.manual_seed(seed)

    lowest_loss_solution = None
    lowest_loss = np.inf
    lowest_loss_obj = np.inf
    lowest_loss_trial = None
    lowest_loss_step = None
    best_valid_solution = None
    best_valid_obj = np.inf
    best_valid_step = None
    best_valid_trial = None
    
    trial_histories = []
    cnt_check = 0

    found = False
    start_t = time()
    for trial_time in range(NUM_RE_TRIALS):
        path_history = []
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution']
            else:
                init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS)).double()
        else:
            init_path = torch.rand((N_WAYPOINTS, robot.dof)).double()
            init_path = init_path * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)

        for step in range(MAXITER):
            opt.zero_grad()
            collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
            cnt_check += len(p) # Counting collision checks
            control_points = robot.fkine(p)
            max_move_cost = torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-1.5**2, min=0).sum()
            joint_limit_cost = (
                torch.clamp(robot.limits[:, 0]-p, min=0) + torch.clamp(p-robot.limits[:, 1], min=0)).sum()
            diff = (control_points[1:]-control_points[:-1]).pow(2).sum()
            constraint_loss = collision_weight * collision_score\
                + max_move_weight * max_move_cost + joint_limit_weight * joint_limit_cost
            objective_loss = dif_weight * diff
            loss = objective_loss + constraint_loss
            loss.backward()
            p.grad[[0, -1]] = 0.0
            opt.step()
            p.data[:, 2] = utils.wrap2pi(p.data[:, 2])
            if history:
                path_history.append(p.data.clone())
            if loss.data.numpy() < lowest_loss:
                lowest_loss = loss.data.numpy()
                lowest_loss_solution = p.data.clone()
                lowest_loss_step = step
                lowest_loss_trial = trial_time
                lowest_loss_obj = objective_loss.data.numpy()
            if constraint_loss <= 1e-2:
                if objective_loss.data.numpy() < best_valid_obj:
                    best_valid_obj = objective_loss.data.numpy()
                    best_valid_solution = p.data.clone()
                    best_valid_step = step
                    best_valid_trial = trial_time
            # if constraint_loss <= 1e-2 or step % (MAXITER/5) == 0 or step == MAXITER-1:
            #     print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
            #         trial_time, step, 
            #         collision_score.item(), collision_weight,
            #         max_move_cost.item(), max_move_weight,
            #         diff.item(), dif_weight,
            #         loss.item()))
            if constraint_loss <= 1e-2 and torch.norm(p.grad) < 1e-4:
                break
        trial_histories.append(path_history)
        
        if best_valid_solution is not None:
            found = True
            break
    end_t = time()
    if not found:
        # print('Did not find a valid solution after {} trials!\
            # Giving the lowest cost solution'.format(NUM_RE_TRIALS))
        solution = lowest_loss_solution
        solution_step = lowest_loss_step
        solution_trial = lowest_loss_trial
        solution_obj = lowest_loss_obj
    else:
        solution = best_valid_solution
        solution_step = best_valid_step
        solution_trial = best_valid_trial
        solution_obj = best_valid_obj
    path_history = trial_histories[solution_trial] # Could be empty when history = false
    if not path_history:
        path_history.append(solution)
    else:
        path_history = path_history[:(solution_step+1)]
    
    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': solution_obj.item(),
        'time': end_t - start_t,
        'success': found,
        'seed': seed,
        'solution': solution.numpy().tolist()
    }
    return rec

def givengrad_traj_optimize(robot, dist_est, start_cfg, target_cfg, options):
    N_WAYPOINTS = options['N_WAYPOINTS'] # 20
    NUM_RE_TRIALS = options['NUM_RE_TRIALS'] # 10
    MAXITER = options['MAXITER'] # 200
    safety_margin = options['safety_margin']

    seed = options['seed']
    torch.manual_seed(seed)

    global cnt_check, obj, max_move_cost, collision_cost, joint_limit_cost, call_cnt
    global var_p_max_move, var_p_collision, var_p_limit, var_p_cost
    global latest_p_max_move, latest_p_collision, latest_p_limit, latest_p_cost
    cnt_check = 0
    call_cnt = 0

    def pre_process(p):
        global var_p
        p = torch.DoubleTensor(p).reshape([-1, robot.dof])
        p[:, 2] = utils.wrap2pi(p[:, 2])
        var_p = torch.cat([init_path[:1], p, init_path[-1:]], dim=0).requires_grad_(True)
        return var_p

    def con_max_move(p):
        global max_move_cost, var_p_max_move, latest_p_max_move
        var_p_max_move = pre_process(p)
        latest_p_max_move = var_p_max_move.data[1:-1].numpy().reshape(-1)
        control_points = robot.fkine(var_p_max_move)
        max_move_cost = -torch.clamp_((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-1.5**2, min=0).sum()
        return max_move_cost.data.numpy()
    def grad_con_max_move(p):
        if all(p == latest_p_max_move):
            var_p_max_move.grad = None
            max_move_cost.backward()
            if var_p_max_move.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_max_move.grad[1:-1].numpy().reshape(-1)
        else:
            raise ValueError('p is not the same as the lastest passed p')

    def con_collision_free(p):
        global cnt_check, collision_cost, var_p_collision, latest_p_collision
        var_p_collision = pre_process(p)
        latest_p_collision = var_p_collision.data[1:-1].numpy().reshape(-1)
        cnt_check += len(p)
        collision_cost = torch.sum(-torch.clamp_(dist_est(var_p_collision[1:-1])-safety_margin, min=0))
        return collision_cost.data.numpy()
    def grad_con_collision_free(p):
        if all(p == latest_p_collision):
            var_p_collision.grad = None
            collision_cost.backward()
            if var_p_collision.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_collision.grad[1:-1].numpy().reshape(-1)
        else:
            raise ValueError('p is not the same as the lastest passed p')

    def con_joint_limit(p):
        global joint_limit_cost, var_p_limit, latest_p_limit
        var_p_limit = pre_process(p)
        latest_p_limit = var_p_limit.data[1:-1].numpy().reshape(-1)
        joint_limit_cost = -torch.sum(torch.clamp_(robot.limits[:, 0]-var_p_limit, min=0)\
             + torch.clamp_(var_p_limit-robot.limits[:, 1], min=0))
        return joint_limit_cost.data.numpy()
    def grad_con_joint_limit(p):
        if all(p == latest_p_limit):
            var_p_collision.grad = None
            joint_limit_cost.backward()
            if var_p_collision.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_collision.grad[1:-1].numpy().reshape(-1)
        else:
            raise ValueError('p is not the same as the lastest passed p')

    def cost(p):
        global obj, var_p_cost, latest_p_cost
        var_p_cost = pre_process(p)
        latest_p_cost = var_p_cost.data[1:-1].numpy().reshape(-1)
        control_points = robot.fkine(var_p_cost)
        obj = (control_points[1:]-control_points[:-1]).pow(2).sum()
        return obj.data.numpy()
    def grad_cost(p):
        if np.allclose(p, latest_p_cost):
            var_p_cost.grad = None
            obj.backward()
            if var_p_cost.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_cost.grad[1:-1].numpy().reshape(-1)
        else:
            print(p, latest_p_cost, np.linalg.norm(p-latest_p_cost))
            raise ValueError('p is not the same as the lastest passed p')

    start_t = time()
    success = False
    for trial_time in range(NUM_RE_TRIALS):
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution']
            else:
                init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS, dtype=np.float64))
        else:
            # init_path = (torch.rand(N_WAYPOINTS, robot.dof, dtype=torch.float64))*np.pi*2-np.pi
            init_path = torch.rand((N_WAYPOINTS, robot.dof)).double()
            init_path = init_path * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        res = minimize(cost, init_path[1:-1].reshape(-1).numpy(), jac=grad_cost,
            method='slsqp',
            constraints=[
                {'fun': con_max_move, 'type': 'ineq', 'jac': grad_con_max_move},
                {'fun': con_collision_free, 'type': 'ineq', 'jac': grad_con_collision_free},
                {'fun': con_joint_limit, 'type': 'ineq', 'jac': grad_con_joint_limit}
            ],
            options={'maxiter': MAXITER, 'disp': False})
        if res.success:
            success = True
            break
    end_t = time()
    res.x = res.x.reshape([-1, robot.dof])
    res.x = pre_process(res.x)
    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': res.fun.item(),
        'time': end_t - start_t,
        'success': success,
        'seed': seed,
        'solution': res.x.data.numpy().tolist()
    }
    return rec

def animation_demo(robot, p, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=None, path_history=None, save_dir=None):
    global opt, start_frame, cnt_down
    FPS = 15

    def init():
        if robot.dof == 2:
            return [link_plot, joint_plot, eff_plot] + cfg_path_plots
        else:
            return link_plot, joint_plot, eff_plot

    def update_traj(i):
        if robot.dof == 2:
            for cfg_path in cfg_path_plots:
                cfg_path.set_data(path_history[i][:, 0], path_history[i][:, 1])
            return cfg_path_plots
        else:
            return link_plot, joint_plot, eff_plot
        
    def plot_robot(q):
        robot_points = robot.fkine(q)[0]
        robot_points = torch.cat([torch.zeros(1, 2), robot_points])
        link_plot.set_data(robot_points[:, 0], robot_points[:, 1])
        joint_plot.set_data(robot_points[:-1, 0], robot_points[:-1, 1])
        eff_plot.set_data(robot_points[-1:, 0], robot_points[-1:, 1])

        return link_plot, joint_plot, eff_plot

    def move_robot(i):
        i = i if i < len(p) else len(p)-1
        with torch.no_grad():
            ret = plot_robot(p[i])
        if robot.dof == 2:
            return list(ret) + cfg_path_plots
        else:
            return ret

    if robot.dof == 2 and path_history:
        UPDATE_STEPS = len(path_history)
        f = lambda i: update_traj(i) if i < UPDATE_STEPS else move_robot(i-UPDATE_STEPS)
        num_frames = UPDATE_STEPS + len(p)
    else:
        f = move_robot
        num_frames=len(p)
    ani = animation.FuncAnimation(
        fig, func=f, 
        frames=num_frames, interval=1000./FPS, 
        blit=True, init_func=init, repeat=False)
    
    if save_dir:
        ani.save(save_dir, fps=FPS)
    else:
        # plt.axis('equal')
        # plt.axis('tight')
        plt.show()

# A function that controls the style of visualization.
def create_plots(robot, obstacles, dist_est, checker):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    if robot.dof > 2:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111) #, projection='3d'
    elif robot.dof == 2:
        # Show C-space at the same time
        num_class = getattr(checker, 'num_class', 1)
        fig = plt.figure(figsize=(3*(num_class + 1), 3 * num_class))
        gs = fig.add_gridspec(num_class, num_class+1)
        ax = fig.add_subplot(gs[:, :-1])
        cfg_path_plots = []

        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
        grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
        score_spline = dist_est(grid_points).reshape(size+[num_class])
        c_axes = []
        with sns.axes_style('ticks'):
            for cat in range(num_class):
                c_ax = fig.add_subplot(gs[cat, -1])

                # score_DiffCo = checker.score(grid_points).reshape(size)
                # score = (torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.max())
                score = score_spline[:, :, cat]
                color_mesh = c_ax.pcolormesh(xx, yy, score, cmap=cmaps[cat], vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                c_support_points = checker.support_points[checker.gains[:, cat] != 0]
                c_ax.scatter(c_support_points[:, 0], c_support_points[:, 1], marker='.', c='black', s=1.5)
                contour_plot = c_ax.contour(xx, yy, score, levels=[-18, -10, 0, 3.5 if cat==0 else 2.5], linewidths=1, alpha=0.4, colors='k') #-1.5, -0.75, 0, 0.3
                ax.clabel(contour_plot, inline=1, fmt='%.1f', fontsize=8)
                # Comment these out if you want colorbars, grad arrows for debugging purposes
                # fig.colorbar(color_mesh, ax=c_ax)
                # sparse_score = score[5:-5:10, 5:-5:10]
                # score_grad_x = -ndimage.sobel(sparse_score.numpy(), axis=1)
                # score_grad_y = -ndimage.sobel(sparse_score.numpy(), axis=0)
                # score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
                # score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
                # score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
                # c_ax.quiver(xx[5:-5:10, 5:-5:10], yy[5:-5:10, 5:-5:10], score_grad_x, score_grad_y, color='red', width=2e-3, headwidth=2, headlength=5)
                # cfg_point = Circle(collision_cfgs[0], radius=0.05, facecolor='orange', edgecolor='black', path_effects=[path_effects.withSimplePatchShadow()])
                # c_ax.add_patch(cfg_point)
                cfg_path, = c_ax.plot([], [], '-o', c='orange', markersize=3)
                cfg_path_plots.append(cfg_path)

                c_ax.set_aspect('equal', adjustable='box')
                # c_ax.axis('equal')
                c_ax.set_xlim(-np.pi, np.pi)
                c_ax.set_ylim(-np.pi, np.pi)
                c_ax.set_xticks([-np.pi, 0, np.pi])
                c_ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'], fontsize=18)
                c_ax.set_yticks([-np.pi, 0, np.pi])
                c_ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'], fontsize=18)

    # Plot ostacles
    # ax.axis('tight')
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xticks([-4, 0, 4])
    # ax.set_yticks([-4, 0, 4])
    ax.tick_params(labelsize=18)
    for obs in obstacles:
        cat = obs[3] if len(obs) >= 4 else 1
        print('{}, cat {}, {}'.format(obs[0], cat, obs))
        if obs[0] == 'circle':
            ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()], color=cmaps[cat](0.5)))
        elif obs[0] == 'rect':
            ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()], 
            color=cmaps[cat](0.5)))
    
    # Placeholder of the robot plot

    
    # trans = ax.transData.transform
    # lw = ((trans((1, robot.link_width))-trans((0,0)))*72/ax.figure.dpi)[1]
    # link_plot, = ax.plot([], [], color='silver', alpha=0.1, lw=lw, solid_capstyle='round', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
    # joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
    # eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)

    if robot.dof > 2:
        return fig, ax
    elif robot.dof == 2:
        return fig, ax, cfg_path_plots
    
def single_plot(robot, path, fig, cfg_path_plots=None, path_history=None, save_dir=None, ax=None):
    from copy import copy
    from matplotlib.lines import Line2D
    import matplotlib as mpl
    points_traj = robot.fkine(path)
    points_traj = torch.cat([torch.zeros(len(path), 1, 2), points_traj], dim=1)
    traj_alpha = 0.3
    ends_alpha = 0.5
    robot_color = 'grey'
    start_robot_color = 'green'
    end_robot_color = 'orange'
    
    robot_traj_patches = []
    for q in path:
        points = robot.fkine(q)[0]
        robot_patch = []
        for p, trans in zip(robot.parts, points):
            if p[0] == 'circle':
                circle_patch = Circle(trans, p[2], color=robot_color, alpha=traj_alpha)
                ax.add_patch(circle_patch) #path_effects=[path_effects.withSimplePatchShadow()], 
                robot_patch.append(circle_patch)
            elif p[0] == 'rect':
                lower_left = torch.FloatTensor([-float(p[2][0])/2, -float(p[2][1])/2])
                R = utils.rot_2d(q[2:3])[0]
                lower_left_position = R@lower_left + trans
                rect_patch = Rectangle([0, 0], p[2][0], p[2][1], color=robot_color, alpha=traj_alpha)
                tf = mpl.transforms.Affine2D().rotate(q[2].item()).translate(*lower_left_position) + ax.transData
                rect_patch.set_transform(tf)
                ax.add_patch(rect_patch) #, path_effects=[path_effects.withSimplePatchShadow()]
                robot_patch.append(rect_patch)
        robot_traj_patches.append(robot_patch)

    # lw = link_plot.get_lw()
    # robot_patches = [ax.plot(points[:, 0], points[:, 1], color='gray', alpha=traj_alpha, lw=lw, solid_capstyle='round')[0] for points in points_traj]
    # joint_traj = [ax.plot(points[:-1, 0], points[:-1, 1], 'o', color='tab:red', alpha=traj_alpha, markersize=lw)[0] for points in points_traj]
    # eff_traj = [ax.plot(points[-1:, 0], points[-1:, 1], 'o', color='black', alpha=traj_alpha, markersize=lw)[0] for points in points_traj]

    for i in [0, -1]:
        for patch in robot_traj_patches[i]:
            patch.set_alpha(ends_alpha)
            # patch.set_path_effects([path_effects.SimpleLineShadow(), path_effects.Normal()])
            # joint_traj[i].set_alpha(ends_alpha)
            # eff_traj[i].set_alpha(ends_alpha)
    for patch in robot_traj_patches[0]:
        patch.set_color(start_robot_color)
    for patch in robot_traj_patches[-1]:
        patch.set_color(end_robot_color)

    # def divide(p): # divide the path into several segments that obeys the wrapping around rule [TODO]
    #     diff = torch.abs(p[:-1]-p[1:])
    #     div_idx = torch.where(diff.max(1) > np.pi)
    #     div_idx = torch.cat([-1, div_idx])
    #     segments = []
    #     for i in range(len(div_idx)-1):
    #         segments.append(p[div_idx[i]+1:div_idx[i+1]+1])
    #     segments.append(p[div_idx[-1]+1:])
    #     for i in range(len(segments)-1):
    #         if torch.sum(torch.abs(segments[i]) > np.pi) == 2:


    # for cfg_path in cfg_path_plots:
    #     cfg_path.set_data(p[:, 0], p[:, 1])

    # ---------Just for making the opening figure------------
    # For a better way wrap around trajectories in angular space
    # see active.py

    # segments = [p[:-3], p[-3:]]
    # d1 = segments[0][-1, 0]-(-np.pi)
    # d2 = np.pi - segments[1][0, 0]
    # dh = segments[1][0, 1] - segments[0][-1, 1]
    # intery = segments[0][-1, 1] + dh/(d1+d2)*d1
    # segments[0] = torch.cat([segments[0], torch.FloatTensor([[-np.pi, intery]])])
    # segments[1] = torch.cat([torch.FloatTensor([[np.pi, intery]]), segments[1]])
    # for cfg_path in cfg_path_plots:
    #     for seg in segments:
    #         cfg_path.axes.plot(seg[:, 0], seg[:, 1], '-o', c='olivedrab', alpha=0.5, markersize=3)
    # ---------------------------------------------

def escape(robot, dist_est, start_cfg):
    N_WAYPOINTS = 20
    UPDATE_STEPS = 200
    safety_margin = -0.3
    lr = 1
    path_history = []
    init_path = start_cfg
    p = init_path.requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)

    for step in range(N_WAYPOINTS):
        if step % 1 == 0:
            path_history.append(p.data.clone())

        opt.zero_grad()
        collision_score = dist_est(p)-safety_margin
        loss = collision_score
        loss.backward()
        opt.step()
        # print()
        p.data[2] = utils.wrap2pi(p.data[2])
        if collision_score <= 1e-4:
            break
    return torch.stack(path_history, dim=0)

# Commented out lines include convenient code for debugging purposes
def main():
    env_name = '10obs_binary_00'

    dataset = torch.load('data/se2_{}.pt'.format(env_name))
    cfgs = dataset['data'].double()
    labels = dataset['label'].double()
    dists = dataset['dist'].double()
    obstacles = dataset['obs']
    # obstacles = [obs+(i, ) for i, obs in enumerate(obstacles)]
    print(obstacles)
    robot = dataset['robot'](*dataset['rparam'])
    train_num = 6000
    fkine = robot.fkine
    checker = DiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    # checker = MultiDiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

    # Check DiffCo test ACC
    test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    assert(test_acc > 0.9)

    fitting_target = 'label' # {label, dist, hypo}
    Epsilon = 1 #0.01
    checker.fit_poly(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target)
    dist_est = checker.rbf_score
    min_score = dist_est(cfgs[train_num:]).min()
    print('MIN_SCORE = {:.6f}'.format(min_score))

    # return # DEBUGGING

    cfg_path_plots = []
    if robot.dof > 2:
        fig, ax, = create_plots(robot, obstacles, dist_est, checker)
    elif robot.dof == 2:
        fig, ax, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)

    

    # Begin optimization
    free_cfgs = cfgs[labels == -1]
    indices = torch.randint(0, len(free_cfgs), (2, ))
    while indices[0] == indices[1]:
        indices = torch.randint(0, len(free_cfgs), (2, ))
    start_cfg = free_cfgs[indices[0]] # torch.zeros(robot.dof, dtype=torch.float32) # 
    target_cfg = free_cfgs[indices[1]] # torch.zeros(robot.dof, dtype=torch.float32) # 

    # collided_cfgs = cfgs[labels == 1]
    # indice = torch.randint(0, len(collided_cfgs), (1,))[0]
    # start_cfg = collided_cfgs[indice] # 
    # print("Start from: ", start_cfg)

    # start_cfg = torch.FloatTensor([2.5, 5, 0])
    # start_cfg[1] = -np.pi/6
    # target_cfg[0] = 3*np.pi/4 #-np.pi/2 # -15*np.pi/16 #  # #  #np.pi# # # 
    # target_cfg[1] = np.pi/5

    ## This is for doing traj optimization
    # p, path_history, num_trial, num_step = traj_optimize(
    #     robot, dist_est, start_cfg, target_cfg, history=True)
    # with open('results/path_se2_{}.json'.format(env_name), 'w') as f:
    #     json.dump(
    #         {
    #             'path': p.data.numpy().tolist(), 
    #             'path_history': [tmp.data.numpy().tolist() for tmp in path_history],
    #             'trial': num_trial,
    #             'step': num_step
    #         },
    #         f, indent=1)
    #     print('Plan recorded in {}'.format(f.name))

    options = {
        'N_WAYPOINTS': 20,
        'NUM_RE_TRIALS': 50,
        'MAXITER': 200,
        'safety_margin': max(1/5*min_score, -1.0),
        'seed': 12345,
        'history': False
    }
    # rec = givengrad_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=options)
    rec = adam_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=options)

    p = torch.FloatTensor(rec['solution'])
    print('Succeeded!' if rec['success'] else 'Not successful')
    with open('results/path_se2_{}.json'.format(env_name), 'w') as f:
        json.dump(
            {
                'path': p.data.numpy().tolist(), 
            },
            f, indent=1)
        print('Plan recorded in {}'.format(f.name))

    ## This for doing the escaping-from-collision experiment
    # p = escape(robot, dist_est, start_cfg)
    # with open('results/esc_se2_{}.json'.format(env_name), 'w') as f:
    #     json.dump({'path': p.data.numpy().tolist(), },f, indent=1)
    #     print('Plan recorded in {}'.format(f.name))
    # with open('results/esc_se2_{}.json'.format(env_name), 'r') as f:
    #     path_dict = json.load(f)
    #     p = torch.FloatTensor(path_dict['path'])
    #     print('Esc plan loaded from {}'.format(f.name))

    ## This is for loading previously computed trajectory
    # with open('results/path_se2_{}.json'.format(env_name), 'r') as f:
    #     path_dict = json.load(f)
    #     p = torch.FloatTensor(path_dict['path'])
    #     path_history = [torch.FloatTensor(shot) for shot in path_dict['path_history']] #[p] #
    
    ## This produces an animation for the trajectory
    # vid_name = None #'results/maual_trajopt_se2_{}_fitting_{}_eps_{}_dif_{}_updates_{}_steps_{}.mp4'.format(
    #     # robot.dof, env_name, fitting_target, Epsilon, dif_weight, UPDATE_STEPS, N_STEPS)
    # if robot.dof == 2:
    #     animation_demo(
    #         robot, p, fig, link_plot, joint_plot, eff_plot, 
    #         cfg_path_plots=cfg_path_plots, path_history=path_history, save_dir=vid_name)
    # elif robot.dof == 7:
    #     animation_demo(robot, p, fig, link_plot, joint_plot, eff_plot, save_dir=vid_name)

    # (Recommended) This produces a single shot of the planned trajectory
    single_plot(robot, p, fig, cfg_path_plots=cfg_path_plots, ax=ax)
    # plt.show()
    # plt.savefig('figs/path_se2_{}.png'.format(env_name), dpi=500)
    plt.savefig('figs/se2_{}_adam_05'.format(env_name), dpi=500) #_equalmargin.png

    # plt.tight_layout()
    # plt.savefig('figs/opening_contourline.png', dpi=500, bbox_inches='tight')
    
    




if __name__ == "__main__":
    main()