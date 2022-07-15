import casadi
import numpy as np

F = np.array([])
B = np.array([])
Q = np.array([])
R = np.array([])
z = x = np.array([])
T = 20
dt = 0.02
prev_u_vec = np.array([(opti.variable(), opti.variable())]*T)
u_vec = np.array([(opti.variable(), opti.variable())]*T)
ref = np.array([0, 0])

# opti.set_initial(x, i[0])
# opti.set_initial(y, i[1])

opti = ca.Opti()

while True:
    u_vec = prev_u_vec

    def traj_cost():
        cost = 0
        for u in u_vec:
            x = F.dot(x) + B.dot(u)
            e = x - ref
            cost += e.transpose() * Q * e + u.transpose() * R * u
        return cost

    # opti.subject_to() 

    opti.minimize(traj_cost())
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000}  # used to be 100
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        # apply u_vec[0]

    except Exception as e:
        print("Did not converge")
        exit(0)

    prev_u_vec = u_vec














'''
def __path_optimizer(imap, traj_raw, fitted_params, stps_tkn, str_pt, end_pt):
        # Docs:
        # http://casadi.sourceforge.net/api/html/dd/dc6/classcasadi_1_1Opti.html

        traj_computed = []
        traj_var = []

        opti = ca.Opti()

        traj_raw.insert(0, str_pt)
        traj_raw.append(end_pt)
        for i in traj_raw:
            x = opti.variable()
            y = opti.variable()
            opti.set_initial(x, i[0])
            opti.set_initial(y, i[1])
            traj_var.append((x, y,))

        opti.subject_to(traj_var[0][0] == str_pt[0])
        opti.subject_to(traj_var[0][1] == str_pt[1])
        opti.subject_to(traj_var[-1][0] == end_pt[0])
        opti.subject_to(traj_var[-1][1] == end_pt[1])

        for i in range(len(traj_var) - 1):
            opti.subject_to(((traj_var[i][0] -
                              traj_var[i -
                                       1][0])**2 +
                             (traj_var[i][1] -
                              traj_var[i -
                                       1][1])**2) == ((traj_var[i][0] -
                                                       traj_var[i +
                                                                1][0])**2 +
                                                      (traj_var[i][1] -
                                                       traj_var[i +
                                                                1][1])**2))

        path_len = 0
        for i in range(len(traj_var)):
            path_len += ((traj_var[i - 1][0] - traj_var[i][0]) **
                         2 + (traj_var[i - 1][1] - traj_var[i][1]) ** 2)**0.5
            opti.subject_to(traj_var[i][0] > 0)
            opti.subject_to(traj_var[i][1] > 0)
            opti.subject_to(traj_var[i][0] < (len(imap) - 1))
            opti.subject_to(traj_var[i][1] < (len(imap) - 1))

        opti.subject_to(path_len < (150-stps_tkn))  # max path length
        opti.subject_to(path_len > (30-stps_tkn))  # prevent clustering

        def traj_cost():
            cost = 0
            for i in traj_var:
                cost += -(self.__loss_func(i, *fitted_params))
            return cost

        opti.minimize(traj_cost())
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000}  # used to be 100
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
            for i in range(len(traj_var)):
                traj_computed.append(
                    (sol.value(
                        traj_var[i][0]), sol.value(
                        traj_var[i][1]),))
        except Exception as e:
            print("Did not converge")
            for i in traj_var:
                traj_computed.append(
                    (opti.debug.value(
                        i[0]), opti.debug.value(
                        i[1]),))
        traj_computed = np.array(traj_computed, dtype=int)

        # Check constraints met
        # for i in range(len(traj_computed) - 1):
        #     print(math.dist(traj_computed[i], traj_computed[i+1]))
        return traj_computed

'''
