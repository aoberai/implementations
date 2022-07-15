#!/usr/bin/env python3

import math
import wpilib
import wpimath.controller
from physics import PhysicsEngine
import time
import casadi as ca
import numpy as np
from scipy.signal import StateSpace
import random


class MPC_Jointed_Arm(wpilib.TimedRobot):
    def robotInit(self) -> None:
        self.encoder = wpilib.Encoder(
            (kEncoderAChannel := 0),
            (kEncoderBChannel := 1))
        self.motor = wpilib.PWMSparkMax((kMotorPort := 0))
        self.joystick = wpilib.XboxController((kJoystickPort := 0))
        # distance per pulse = (angle per revolution) / (pulses per revolution)
        # = (2 * PI rads) / (4096 pulses)
        self.kArmEncoderDistPerPulse = 2.0 * math.pi / 4096.0
        self.encoder.setDistancePerPulse(self.kArmEncoderDistPerPulse)

    def teleopInit(self) -> None:
        self.kKt = wpimath.system.plant.DCMotor.vex775Pro().Kt
        self.kKv = wpimath.system.plant.DCMotor.vex775Pro().Kv
        self.kR = wpimath.system.plant.DCMotor.vex775Pro().R
        self.kG = 1200
        self.kL = 0.762
        self.kM = 5
        self.kJ = wpilib.simulation.SingleJointedArmSim.estimateMOI(self.kL, self.kM)
        print(self.kKt, self.kKv, self.kR, self.kJ)
        self.kGr = 9.81
        self.F_cnt = np.array(
            [[0, 1], [0, -self.kG**2 * self.kKt / (self.kKv * self.kR * self.kJ)]])
        self.B_cnt = np.array(
            [[0], [self.kG * self.kKt / (self.kR * self.kJ)]])
        self.C = np.array([[1, 0]])
        self.D = np.array([[0]])
        self.kDt = 0.02
        tmp = StateSpace(
            self.F_cnt,
            self.B_cnt,
            self.C,
            self.D).to_discrete(
            self.kDt)
        self.F = tmp.A
        self.B = tmp.B
        self.C = tmp.C
        self.D = tmp.D
        self.Q = np.array([[1.0, 0], [0, 0.1]])
        self.R = np.array([0.01])
        self.T = 20
        self.opti = ca.Opti()
        self.prev_u_vec = []
        self.u_vec = []
        for i in range(self.T):
            self.prev_u_vec.append(self.opti.variable())
            self.u_vec.append(self.opti.variable())
        self.ref = np.array([[math.radians(180)], [0]])

    def teleopPeriodic(self) -> None:
        self.u_vec = self.prev_u_vec.copy()
        # TODO: add in gravity feedforward

        def convergence_cost():
            cost = 0
            x = np.array([[self.encoder.getDistance()], [self.encoder.getRate()]])
            print("Error:", self.C.dot(self.ref) - self.C.dot(x))
            for u in self.u_vec:
                # grav_ff = self.kL * self.R * self.kM * self.kGr * math.cos(self.encoder.getDistance())/ (2 * self.kG * self.kKt)
                # print("Grav_ff: ", grav_ff)
                x = self.F.dot(x) + self.B.dot(np.array([u]))
                e = self.ref - x
                cost += (e.transpose() * self.Q * e +
                         np.array(u).transpose() * self.R * np.array(u))[0][0]
            return cost

        self.opti.minimize(convergence_cost())
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000}  # used to be 100
        self.opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = self.opti.solve()
            u_vec_val = []
            for i in range(len(self.u_vec)):
                u_vec_val.append(sol.value(self.u_vec[i][0]))

            xu_vec = []
            x = np.array([[self.encoder.getDistance()], [self.encoder.getRate()]])
            for u in u_vec_val:
                # print(np.shape(self.F), np.shape(self.x), np.shape(self.B), np.shape(np.array(u)))
                x = self.F.dot(x) + self.B.dot(np.array(u))
                xu_vec.append((x.tolist(), u,))
            print("XU's:", xu_vec)
            # grav_ff = self.kL * self.R * self.kM * self.kGr * math.cos(self.encoder.getDistance())/ (2 * self.kG * self.kKt)
            self.motor.set(u_vec_val[0])
            # self.motor.set(grav_ff)

        except Exception as e:
            print("Did not converge")
            print(e)
            exit(0)

        del self.u_vec[0]
        self.u_vec.append(self.opti.variable())
        self.prev_u_vec = self.u_vec

    def disabledPeriodic(self) -> None:
            # motor is off
            self.motor.set(0)


if __name__ == "__main__":
    wpilib.run(MPC_Jointed_Arm)
