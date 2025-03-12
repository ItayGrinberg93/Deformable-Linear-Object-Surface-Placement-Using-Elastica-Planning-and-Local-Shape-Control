import numpy as np
import help_funcs as hf


class Node:
    """
    Attributes:
    -----------
    target - the cornet desired location of the navigation
    X0 - The x pose free edge of the object
    Y0 - The y pose free edge of the object
    Z0 - The z pose free edge of the object
    phi_0 - The angle of the free edge of the object
    theta - The angle of the DLO around Z in the world axis
    mu - Modul elliptical related to the maximum tangent direction relative to elastica axis
    S0 - Phase parameter of s
    L_gal - The full period length of the elastica
    l - The part of the DLO which lies on
    father - The previous node
    alpha - Wighted vector
    L - The length of the DLO
    XL - The x pose of EE
    YL - The y pose of EE
    ZL - The z pose of EE
    phi_L - The angle of the EE
    G - The arc-length cost
    h - The heuristic cost
    F - The total cost
    jointsPose - The joint positions the robot gets
    """

    def __init__(self, target, X0, Y0, Z0, phi_0, theta, mu, S0, L_gal, l, father, alpha, L, direction, mode):
        self.X0 = X0
        self.Y0 = Y0
        self.Z0 = Z0
        self.phi_0 = phi_0
        self.mode = mode
        if direction == "right":
            [XL, ZL, phi_L] = hf.get_shape(l, Z0, phi_0, mu, L_gal, S0, L)
        elif direction == "left":
            [XL, ZL, phi_L] = hf.get_shape(-l, Z0, phi_0, mu, L_gal, S0, L)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.mu = mu
        self.S0 = S0
        self.L_gal = L_gal
        XY = np.array([[X0], [Y0]]) + np.dot(R, np.array([[XL], [0]]))
        self.XL = XY[0]
        self.YL = XY[1]
        self.ZL = ZL
        self.phi_L = phi_L - np.pi
        self.l = l
        self.theta = theta
        if not father:
            self.father = []
            self.G = 0
        else:
            self.father = father
            self.G = father.G + hf.get_dist(self, father, alpha, mode)
        w = 0.88
        self.h = hf.get_dist(self, target, alpha, mode)
        self.F = (1 - w)*self.G + w*self.h
        self.jointsPose = []


class Target:
    """
    Attributes:
    -----------
    X0 - The x pose free edge of the object
    Y0 - The y pose free edge of the object
    Z0 - The z pose free edge of the object
    phi_0 - The tangent angle of the free edge of the object
    theta - The angle of the DLO around Z in the world axis
    mu - Modul elliptical related to the maximum tangent direction relative to elastica axis
    S0 - Phase parameter of s
    L_gal - The full period length of the elastica
    l - The part of the DLO which lies on
    father - The previous node
    alpha - Wighted vector
    L - The length of the DLO
    XL - The x pose of EE
    YL - The y pose of EE
    ZL - The z pose of EE
    phi_L - The angle of the EE
    """
    def __init__(self, X0, Y0, Z0, phi_0, theta, mu, S0, L_gal, l, L, direction):
        self.X0 = X0
        self.Y0 = Y0
        self.Z0 = Z0
        self.phi_0 = phi_0
        if direction == "right":
            [XL, ZL, phi_L] = hf.get_shape(l, Z0, phi_0, mu, L_gal, S0, L)
        elif direction == "left":
            [XL, ZL, phi_L] = hf.get_shape(-l, Z0, phi_0, mu, L_gal, S0, L)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.mu = mu
        self.S0 = S0
        self.L_gal = L_gal
        XY = np.array([[X0], [Y0]]) + np.dot(R, np.array([[XL], [0]]))
        self.XL = XY[0]
        self.YL = XY[1]
        self.ZL = ZL
        self.phi_L = phi_L - np.pi
        self.l = l
        self.theta = theta
        self.father = []


class Bound:
    """
    Attributes:
    ----------
    L - The length of the DLO
    direction - The direction of the displacement
    gamma - Friction parameter
    x_max - The upper limit of x pose
    x_min - The lower limit of x pose
    y_max - The upper limit of y pose
    y_min - The lower limit of y pose
    z_max - The upper limit of height
    z_min - The lower limit of height
    theta_max - The upper limit of theta
    theta_min - The lower limit of theta
    dx - The step in X0
    dy - The step in Y0
    dz - The step in Z0
    d_phi_0 - The step in phi_0
    d_theta - The step in theta
    dmu - The step in mu
    dL_gal - The step in L_gal
    dl - The step in l
    alpha - Wight vector
    eps - Max error
    L_gal_max - The upper limit of L_gal
    """

    def __init__(self, L, direction, gamma, x_max, x_min, y_max, y_min, z_max, z_min, theta_max, theta_min, dx, dy, dz,
                 d_phi_0, d_theta, dmu, dL_gal, dl, alpha, L_gal_max):

        self.L = L
        self.displacment_diraction = direction
        self.gamma = gamma
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.z_max = z_max
        self.z_min = z_min
        if direction == "right":
            self.min_phi_0 = 0
            self.max_phi_0 = np.pi / 2 + gamma
        elif direction == "left":
            self.min_phi_0 = np.pi / 2 - gamma
            self.max_phi_0 = np.pi
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.max_mu = 1
        self.min_mu = 1 - 2*0.908**2
        self.min_l = 0
        self.max_l = 0.9*L
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.d_phi_0 = d_phi_0
        self.d_theta = d_theta
        self.dmu = dmu
        self.dL_gal = dL_gal
        self.dl = dl
        self.alpha = alpha
        self.L_gal_max = L_gal_max


class JointNode:
    """
    Attributes:
    -----------
    XL - The x pose of EE
    YL - The y pose of EE
    ZL - The z pose of EE
    phi_L - The tangent angle of DLO where the EE
    theta - The angle of the DLO around Z in the world axis
    father - The previous node
    jointsPose - The joint positions the robot gets
    """
    def __init__(self, XL, YL, ZL, phi_L, theta, father):
        self.XL = XL
        self.YL = YL
        self.ZL = ZL
        self.phi_L = phi_L
        self.theta = theta
        self.father = father
        self.jointsPose = []




