# This script is the whole process of displacement

import numpy as np
import utils as hf
import node_class


def displacement(start_1, target_1, target_2, target_3, L, mu_fric, direction, h_table, alpha, urdf, q_start,mode=1):
    # define the LDO and environment parameters
    gamma = np.arctan(mu_fric)

    Bound = node_class.Bound(L=L, direction=direction, gamma=gamma, x_max=2, x_min=-2, y_max=2, y_min=-2, z_max=4*L,
                             z_min=h_table, theta_max=np.pi, theta_min=-np.pi, dx=0.01, dy=0.01, dz=0.01*L,
                             d_phi_0=2*np.pi/180, d_theta=2*np.pi/180, dmu=0.01, dL_gal=0.01*L, dl=0.01*L,
                             alpha=alpha, L_gal_max=10*L)
    Bound.urdf = urdf

    # get path for sub-problem 1
    if mode == 1:
        path_1 = A_star_displacement(start_1, target_1, Bound, 1, q_start)
        path_1.pop(-1)
        if not path_1:
            return False
        mode = 2
    # get path for sub-problem 2
    start_2 = node_class.Node(target_2, target_1.X0, target_1.Y0, target_1.Z0, target_1.phi_0, target_1.theta,
                              target_1.mu, target_1.S0, target_1.L_gal, target_1.l, [], alpha, L, direction, 2)
    if mode == 2:
        path_2 = A_star_displacement(start_2, target_2, Bound, 2, path_1[-1].jointsPose)
        path_2.pop(-1)
        if not path_2:
            return False
        mode = 3

    # get path for sub-problem 3
    q_start = path_2[-1].jointsPose
    start_3 = node_class.Node(target_3, target_2.X0, target_2.Y0, target_2.Z0, target_2.phi_0, target_2.theta,
                              target_2.mu, target_2.S0, target_2.L_gal, target_2.l, [], alpha, L, direction, 3)
    if mode == 3:
        path_3 = A_star_displacement(start_3, target_3, Bound, 3, path_2[-1].jointsPose)
        path_3.pop(-1)
        if not path_3:
            return False
        mode = 1
    return [path_1, path_2, path_3]


def A_star_displacement(start, target, Bound, mode, q_start):
    PATH = []
    flag = 0
    CHECK = hf.check_node_disp(start, Bound, mode, q_start) and hf.check_node_disp(target, Bound, mode, q_start)
    # start navigation
    Open = [start]
    Close = []
    if CHECK:
        while Open:
            x_best = Open[-1]
            Close.append(x_best)
            Open.pop(-1)
            if hf.check_reach_target(x_best, target, Bound):
                x_best = node_class.Node(target, target.X0, target.Y0, target.Z0, target.phi_0, target.theta,
                                         target.mu, target.S0, target.L_gal, target.l, x_best, Bound.alpha,
                                         Bound.L, Bound.displacment_diraction, mode)
                flag = 1
                break
            Open, Close = hf.get_star(x_best, Open, Close, Bound, target, mode, q_start) # define this function
    else:
        print("there is no valid path for sub-problem", mode)
    if flag:
        path = x_best
        while path:
            PATH.append(path)
            path = path.father
        PATH.reverse()
        return PATH
    else:
        print("there is no valid path fo sub-problem", mode)

