from scipy import special
import node_class
import pinocchio as pin
import numpy as np
import sys
from numpy.linalg import norm, solve


def get_shape(X0, Y0, PHI0, mu, L_gal, s0, s):
    """
    What the function do
    ____________________
    The function give us the elastica shape of the cable

    Parameters
    ----------
    X0 - initial X
    Y0 - initial Y
    PHI0 - initial phi
    mu - modul elliptical related to the maximum tangent direction relative to elastica axis
    L_gal - the full period length of the elastica
    s0 - phase parameter of s
    s - arclength , the cable length between 0 to L

    Returns
    -------
    elastica shape defind by [x,y,phi]
    """
    # elastica parameters:
    k = np.sqrt((1 - mu) / 2)
    l = (4 * special.ellipk(k ** 2) / L_gal) ** 2
    a = np.sqrt(2 * (1 - mu) / l)
    # epsilon:
    (sn, cn, dn, am) = special.ellipj(np.sqrt(l) * (s0 + s), k ** 2)
    jaceps = special.ellipeinc(am, k ** 2)
    (sn1, cn1, dn1, am1) = special.ellipj(np.sqrt(l) * s0, k ** 2)
    jaceps0 = special.ellipeinc(am1, k ** 2)
    # elastic shape before trans:
    x_tilda = (2 / np.sqrt(l)) * (jaceps - jaceps0) - s
    y_tilda = a * (cn - cn1)
    phi_tilda = -2 * np.arcsin(k * sn) + 2 * np.arcsin(k * sn1)
    Psi = PHI0 + 2 * np.arcsin(k * sn1)
    # shape after trans:
    x = X0 + np.cos(Psi) * x_tilda - np.sin(Psi) * y_tilda
    y = Y0 + np.sin(Psi) * x_tilda + np.cos(Psi) * y_tilda
    phi = PHI0 + phi_tilda
    return x, y, phi


def get_dist(node, father, alpha):
    """
    :param node: node on the search graph
    :param father: the previous node
    :param alpha:  wight vector
    :return: norm
    """
    dist = np.sqrt(
        alpha * (node.phi_0 - father.phi_0)**2 +
        alpha * (node.theta - father.theta)**2 +
        (node.X0 - father.X0)**2 +
        (node.Y0 - father.Y0)**2 +
        (node.Z0 - father.Z0)**2 +
        (node.mu - father.mu)**2 +
        (node.L_gal - father.L_gal)**2 +
        (node.l - father.l)**2
    )
    return dist


def check_reach_target(node, target, bound):
    """
    :return: check if the shape is in the target position
    """
    flag = 0
    if np.abs(node.X0 - target.X0) <= bound.dx and np.abs(node.Y0 - target.Y0) <= bound.dy and \
        np.abs(node.Z0 - target.Z0) <= bound.dz and np.abs(node.phi_0 - target.phi_0) <= bound.d_phi_0 and \
        np.abs(node.mu - target.mu) <= bound.dmu and np.abs(node.L_gal - target.L_gal) <= bound.dL_gal and \
            np.abs(node.theta - target.theta) <= bound.d_theta and np.abs(node.l - target.l) <= bound.dl:
        flag = 1
    return flag


def check_in_list(neighbor, List, bound):
    """
    What the function do
    ____________________
    The function checking if a node is in the list

    Returns
    -------
    if True: the index
    if False: bool
    """
    if not List:
        return False
    for n in List:
        flag = np.abs(neighbor.X0 - n.X0) <= bound.dx/2 and np.abs(neighbor.Y0 - n.Y0) <= bound.dy/2 and \
            np.abs(neighbor.Z0 - n.Z0) <= bound.dz/2 and np.abs(neighbor.phi_0 - n.phi_0) <= bound.d_phi_0/2 and \
            np.abs(neighbor.theta - n.theta) <= bound.d_theta/2 and np.abs(neighbor.mu - n.mu) <= bound.dmu/2 and \
            np.abs(neighbor.L_gal - n.L_gal) <= bound.dL_gal/2 and np.abs(neighbor.l - n.l) <= bound.dl/2
        if flag:
            return List.index(n)
    return False


def where_to_put_neighbor(neighbor, open_list):
    """
    What the function do
    ____________________
    Sorting the array with a new node

    Returns
    -------
    Sorting list
    """
    if not open_list: return 0
    index_a = 0
    index_b = len(open_list)
    mid = 0
    while index_a < index_b:
        mid = round((index_a + index_b) / 2)
        if mid > (index_a + index_b) / 2:
            mid = mid - 1
        if open_list[mid].F >= neighbor.F:
            index_a = mid + 1
        else:
            index_b = mid - 1
    return index_a


def get_neighbor(neighbor, open_list, close_list, bound, mode, q_start):
    """
    What the function do
    ____________________
    The function decides whether the current neighbor is acceptable
    and put it in the right place in the array

    Parameters
    ----------
    neighbor
    open_list
    close_list
    bound
    mode

    Returns
    -------
    Sorted open and close lists
    """
    if check_node_disp(neighbor, bound, mode, q_start):
        if not check_in_list(neighbor, close_list, bound):
            index = check_in_list(neighbor, open_list, bound)
            if index:
                if open_list[index].F >= neighbor.F:
                    open_list.remove(open_list[index])
                    index = where_to_put_neighbor(neighbor, open_list)
                    open_list.insert(index, neighbor)
            else:
                index = where_to_put_neighbor(neighbor, open_list)
                open_list.insert(index, neighbor)
    return open_list, close_list


def get_star(x_best, open_list, close_list, bound, target, mode, q_start):
    """
    What the function do
    ____________________

    open all the neighbors of the current node

    Parameters
    ----------
    open_list
    close_list
    x_best
    target
    bound
    mode - indicate related sub-problem 
    q_start - the joints vector of the robot in the start configuration

    Returns
    -------
    Sorted open and close lists
    """
    if bound.displacment_diraction == "right":
        C = 1 / 4
    elif bound.displacment_diraction == "left":
        C = 3 / 4
    if mode == 1:
        n1 = node_class.Node(target, x_best.X0 + bound.dx, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n2 = node_class.Node(target, x_best.X0 - bound.dx, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction,mode)
        n3 = node_class.Node(target, x_best.X0, x_best.Y0 + bound.dy, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction,mode)
        n4 = node_class.Node(target, x_best.X0, x_best.Y0 - bound.dy, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction,mode)
        n5 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0 + bound.dz, x_best.phi_0,
                             x_best.theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n6 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0 - bound.dz, x_best.phi_0,
                             x_best.theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n7 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta + bound.d_theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n8 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta - bound.d_theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        neighbors = [n1, n2, n3, n4, n5, n6, n7, n8]
    elif mode == 2:
        n1 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0 + bound.d_phi_0,
                             x_best.theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n2 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0 - bound.d_phi_0,
                             x_best.theta, x_best.mu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n3 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu + bound.dmu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n4 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu - bound.dmu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n5 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, C*(x_best.L_gal + bound.dL_gal), x_best.L_gal + bound.dL_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n6 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, C*(x_best.L_gal - bound.dL_gal), x_best.L_gal - bound.dL_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        neighbors = [n1, n2, n3, n4, n5, n6]
    elif mode == 3:
        n1 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu + bound.dmu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n2 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu - bound.dmu, x_best.S0, x_best.L_gal, x_best.l,
                             x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n3 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, C * (x_best.L_gal + bound.dL_gal), x_best.L_gal + bound.dL_gal,
                             x_best.l, x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n4 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, C * (x_best.L_gal - bound.dL_gal), x_best.L_gal - bound.dL_gal,
                             x_best.l, x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n5 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, C * (x_best.L_gal - bound.dl), x_best.L_gal - bound.dl,
                             x_best.l + bound.dl, x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        n6 = node_class.Node(target, x_best.X0, x_best.Y0, x_best.Z0, x_best.phi_0,
                             x_best.theta, x_best.mu, C * (x_best.L_gal + bound.dl), x_best.L_gal + bound.dl,
                             x_best.l - bound.dl, x_best, bound.L, bound.alpha, bound.displacment_diraction, mode)
        neighbors = [n1, n2, n3, n4, n5, n6]
    for n in neighbors:
        open_list, close_list = get_neighbor(n, open_list, close_list, bound, mode, q_start)
    return open_list, close_list


def check_node_disp(node, Bound, mode, q_start):
    """
     What the function do
    ____________________
    For each mode the function check if the node is the C space

    Parameters
    ----------
    node
    Bound
    mode

    Returns
    -------
    bool: Ture/False
    """
    if mode == 1:
        check_x = Bound.x_max >= node.X0 >= Bound.x_min
        check_y = Bound.y_max >= node.Y0 >= Bound.y_min
        check_z = Bound.z_max >= node.Z0 >= Bound.z_min
        check_theta = Bound.theta_max >= node.theta >= Bound.theta_min
        check = check_x and check_y and check_z and check_theta
    elif mode == 2:
        check_phi = Bound.max_phi_0 >= node.phi_0 >= Bound.min_phi_0
        check_mu = Bound.max_mu >= node.mu >= Bound.min_mu
        check_L_gal = Bound.L_gal_max >= node.L_gal >= 1.86 * (Bound.L - node.l)  # stability criteria
        if Bound.displacment_diraction == 'right':  # friction cone check
            check_fric = np.pi / 2 + Bound.gamma >= node.phi_0 + np.arccos(node.mu) >= np.pi / 2 - Bound.gamma
        elif Bound.displacment_diraction == 'left':
            check_fric = np.pi / 2 + Bound.gamma >= node.phi_0 - np.arccos(node.mu) >= np.pi / 2 - Bound.gamma
        check = check_phi and check_mu and check_L_gal and check_fric
    elif mode == 3:
        check_mu = Bound.max_mu >= node.mu >= Bound.min_mu
        check_L_gal = Bound.L_gal_max >= node.L_gal >= 1.86 * (Bound.L - node.l)  # stability criteria
        check_l = Bound.min_l <= node.l <= Bound.max_l
        check_fric = np.pi / 2 + Bound.gamma >= np.arccos(node.mu) >= np.pi / 2 - Bound.gamma
        check = check_mu and check_L_gal and check_l and check_fric
    else:
        print("somthing straing!")
        check = 0
    if check:
        check, q = IKFromURDF(Bound.urdf, node, q_start)
        node.jointsPose = q
    return check


def get_smooth_path_displacement(path, skip, t, L, direction, urdf, q_start):
    """
    What the function do
    ____________________
    after planning the path the function smooth it,
    in order for the robot movement to be smoother.

    Parameters
    ----------
    path
    skip
    t
    L
    direction
    urdf

    Returns
    -------
    smooth path
    """
    parameters = []
    pathsmooth = []
    for i in np.arange(0, len(path), skip):
        parameters.append([
            path[i].mu, path[i].S0, path[i].L_gal, path[i].l,
            path[i].phi_0, path[i].theta, path[i].X0, path[i].Y0, path[i].Z0
        ])

    if i != len(path) - 1:
        parameters.append([
            path[-1].mu, path[-1].S0, path[-1].L_gal, path[-1].l,
            path[-1].phi_0, path[-1].theta, path[-1].X0, path[-1].Y0, path[-1].Z0
        ])

    xs = []
    ys = []
    zs = []
    phi_l_s = []
    theta_s = []
    ls = []
    q = []
    Node = [[]]

    if direction == 'right':
        C = 1
    else:
        C = -1

    parameters = np.array(parameters)

    # TODO - check smooth function
    for i in range(1, len(parameters)):
        for j in range(len(t)):
            line = parameters[i - 1] + t[j] * (parameters[i] - parameters[i - 1])
            temp = [line[7]-C * line[3], line[8], line[4], line[2], line[1], line[0], L, line[3]] # smooth path node
            pathsmooth.append(temp)
            xt, zt, phi_t = get_shape(C * line[3], line[8], line[4], line[0], line[2], line[1], np.linspace(0, L - line[3], 50))
            R = np.array([
                [np.cos(line[5]), -np.sin(line[5])],
                [np.sin(line[5]), np.cos(line[5])]
            ])
            XY = np.dot(R, np.vstack([xt, np.zeros_like(xt)]))
            xs.append(XY[0, :] + line[6])
            ys.append(XY[1, :] + line[7])
            zs.append(zt)
            phi_l_s.append(phi_t[-1])
            theta_s.append(line[5])
            ls.append(line[3])
            node = node_class.JointNode(XL=xs[-1][-1], YL=ys[-1][-1], ZL=zs[-1][-1], phi_L=phi_l_s[-1],
                                        theta=line[5], father=Node[-1])
            check, q_temp = IKFromURDF(urdf, node, q_start)
            q.append(q_temp.tolist())
            node.jointsPose = q_temp
            Node.append(node)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    phi_l_s = np.array(phi_l_s)
    theta_s = np.array(theta_s)
    ls = np.array(ls)

    return xs, ys, zs, phi_l_s, theta_s, ls, q, pathsmooth


def get_gripper_path(x, y, z, phi_L, theta):
    """
    What the function do
    ____________________

    Parameters
    ----------
    x
    y
    z
    phi_L
    theta

    Returns
    -------

    """
    R = []
    P = np.column_stack((x, y, z))
    gripper_pose = []
    for i in range(len(phi_L)):
        Rz = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]), 0],
            [np.sin(theta[i]), np.cos(theta[i]), 0],
            [0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(phi_L[i] + np.pi / 2), 0, -np.sin(phi_L[i] + np.pi / 2)],
            [0, 1, 0],
            [np.sin(phi_L[i] + np.pi / 2), 0, np.cos(phi_L[i] + np.pi / 2)]
        ])

        Ri = np.dot(Rz, Ry)
        R.append(Ri)

    return P, R


def IKFromURDF(urdf:str, node, q_start):
    """
    What the function do
    ____________________
    compute the IK based on the urdf

    Parameters
    ----------
    urdf
    node

    Returns
    -------
    if valid: Ture and the joint position
    if not valid: false
    """
    # You should change here to set up your own URDF file or just pass it as an argument of this example.
    urdf_filename = (
        urdf
        if len(sys.argv) < 2
        else sys.argv[1]
    )

    # Load the urdf model
    model = pin.buildModelFromUrdf(urdf_filename)
    # print("model name: " + model.name)

    # Create data required by the algorithms
    data = model.createData()
    JOINT_ID = model.njoints - 1
    # Sample a start configuration
    P, R = get_gripper_path(node.XL, node.YL, node.ZL, [node.phi_L], [node.theta])
    oMdes = pin.SE3(R[0], np.array(P[0]))
    if node.father:
        q = node.father.jointsPose
    else:
        q = q_start # need to get from the robot joint the initial position
    # print("q: %s" % q.T)

    eps = 1e-4
    IT_MAX = 1000
    DT = 1e-1
    damp = 1e-12
    i = 0

    while True:
        pin.forwardKinematics(model, data, q)
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        err = pin.log(iMd).vector  # in joint frame
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pin.computeJointJacobian(model, data, q, JOINT_ID)  # in joint frame
        J = -np.dot(pin.Jlog6(iMd.inverse()), J)
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(model, q, v * DT)
        i += 1

    if success:
        return [True, q]

    else:
        return False




