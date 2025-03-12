import time
import numpy as np
import matplotlib.pyplot as plt
import utils as hf
import node_class
import displacement as dis
import pinocchio as pin
import sys
from scipy.spatial.transform import Rotation
from angle_transformation import Euler2Robot


def displacement_planner(recoverymode = 0,mode = 1):
    # define LDO parameters, environment parameters and direction displacement
    urdf ="./robots_urdf/ur5e/ur5e.urdf" # path of the URDF
    # urdf = "./robots_urdf/iiwa14_with_ft_and_dlr_gripper (copy)/iiwa14_with_ft_and_dlr_gripper.urdf" # path of the URDF
    # urdf = # path of the URDF

    # urdf = ik.chain.Chain.from_urdf_file('URDF/UR5.URDF')
    urdf_filename = (
            urdf
            if len(sys.argv) < 2
            else sys.argv[1]
        )
    model = pin.buildModelFromUrdf(urdf_filename)
    q_start = np.array([50.74, -94.84, 118.95, -114.01, -89.59, -41.81]) #pin.neutral(model) # [50.74, -94.84, 118.95, -114.01, -89.59, -41.81] q_start should be input parameter
    # q_start = np.array([50.74, -94.84, 118.95, -114.01, -89.59, -41.81,0,0,0,0,0,0]) #pin.neutral(model) # [50.74, -94.84, 118.95, -114.01, -89.59, -41.81] q_start should be input parameter


    direction = "right"
    L = 0.142
    mu_fric = 0.1
    h_table = -0.015
    h_tray = 0.006
    alpha = 0.6
    if direction == "right":
        C = 1/4
        phi_0_T2 = 0
    elif direction == "left":
        C = 3/4
        phi_0_T2 = np.pi
    # define start and target for sub-problem 1
    X0_S1 = 0.4
    Y0_S1 = -0.2
    Z0_S1 = h_table + h_tray
    theta_S1 = np.pi/2
    X0_T1 = -0.19004
    Y0_T1 = -0.38931
    Z0_T1 = h_table + h_tray
    theta_T1 = 0
    L_gal_S = 2*L
    target_1 = node_class.Target(X0_T1, Y0_T1, Z0_T1, np.pi/2, theta_T1, 1, C*L_gal_S, L_gal_S, 0, L, direction)
    start_1 = node_class.Node(target=target_1, X0=X0_S1, Y0=Y0_S1, Z0=Z0_S1, phi_0=np.pi/2, theta=theta_S1, mu=1, S0=C*L_gal_S, L_gal=L_gal_S, l=0, father=[], alpha=alpha, L=L, direction=direction,mode=1)

    # define target for sub-problem 2
    mu_T2 = 0.02
    L_gal_T2 = 2.5*L
    target_2 = node_class.Target(target_1.X0, target_1.Y0, target_1.Z0, phi_0_T2, target_1.theta, mu_T2, C*L_gal_T2, L_gal_T2, 0, L, direction)

    # define target for sub-problem 3
    mu_T3 = 0.02
    l_T3 = 0.45*L
    L_gal_T3 = 4.5*(L - l_T3)
    target_3 = node_class.Target(target_2.X0, target_2.Y0, target_2.Z0, phi_0_T2, target_2.theta, mu_T3, C*L_gal_T3, L_gal_T3, l_T3, L, direction)

    # get full path
    t_start = time.time()
    if recoverymode == 1:
        path = dis.displacement(start_1, target_1, target_2, target_3, L, mu_fric, direction, h_table, alpha, urdf, q_start,mode)
    else:
        path = dis.displacement(start_1, target_1, target_2, target_3, L, mu_fric, direction, h_table, alpha, urdf, q_start,mode)

    if path:
        path_1, path_2, path_3 = path

        # check displacement
        skip = 8
        num_t = 8
        t = np.linspace(0, 1, num_t)
        s = np.linspace(0, 1, num_t)

        x1, y1, z1, phi_l_1, theta_1, l1, q1, pathsmooth1 = hf.get_smooth_path_displacement(path_1, skip, t, L, direction, urdf, path_1[0].jointsPose)
        x2, y2, z2, phi_l_2, theta_2, l2, q2, pathsmooth2 = hf.get_smooth_path_displacement(path_2, skip, t, L, direction, urdf, path_2[0].jointsPose)
        x3, y3, z3, phi_l_3, theta_3, l3, q3, pathsmooth3 = hf.get_smooth_path_displacement(path_3, skip, t, L, direction, urdf, path_3[0].jointsPose)

        t_end = time.time()
        ExecutionTime1_2_3 = t_end - t_start
        print("Execution Time =", ExecutionTime1_2_3)

        P1, R1 = hf.get_gripper_path(x1[:, -1], y1[:, -1], z1[:, -1], phi_l_1, theta_1)
        P2, R2 = hf.get_gripper_path(x2[:, -1], y2[:, -1], z2[:, -1], phi_l_2, theta_2)
        P3, R3 = hf.get_gripper_path(x3[:, -1], y3[:, -1], z3[:, -1], phi_l_3, theta_3)

        # make x,y z to np.array
        x = np.array(x1.tolist() + x2.tolist() + x3.tolist())
        y = np.array(y1.tolist() + y2.tolist() + y3.tolist())
        z = np.array(z1.tolist() + z2.tolist() + z3.tolist())

        # check by graphs
        path = path_1 + path_2 + path_3
        gripper_P = np.array(P1.tolist() + P2.tolist() + P3.tolist())
        gripper_R = R1 + R2 + R3
        robot_joint_pose = q1 + q2 + q3
        len1 = len(q1)
        len2 = len(q2)
        len3 = len(q3)
        ### transform the matrix to euler angles
        gripper_pose = []
        for i in range(len(gripper_P)):
            r = Rotation.from_matrix(gripper_R[i])
            angles = np.deg2rad(r.as_euler("zyx", degrees=True))
            angles = Euler2Robot(angles)
            temp = np.array([gripper_P[i][0], gripper_P[i][1], gripper_P[i][2], angles[0], angles[1], angles[2]])
            gripper_pose.append(temp)
        E_params = []
        path123 = pathsmooth1 + pathsmooth2 + pathsmooth3
        phi_l = list(phi_l_1) + list(phi_l_2) + list(phi_l_3)
        for i in range(len(path123)):
            n = path123[i]
            temp = [n[0]+n[7], n[1], n[2], n[3], n[4], n[5], n[6], n[7], gripper_P[i][0],  gripper_P[i][2], phi_l[i]]
            E_params.append(temp)

        return gripper_pose, len1, len2, len3, E_params
        # if direction == "right":
        #     a = 1
        # elif direction == "left":
        #     a = -1
        #     phi_0_T2 = np.pi
        #
        # for n in path:
        #     plt.figure('1')
        #     [xs, zs, phi_s] = hf.get_shape(n.X0 + a*n.l, n.Z0, n.phi_0, n.mu, n.L_gal, n.S0, np.linspace(0, L-n.l))
        #     plt.plot(xs, zs)
        #     plt.grid()
        # plt.show()
        # return gripper_pose, len1, len2, len3, E_params







