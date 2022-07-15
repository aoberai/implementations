# Sim viz from robot py

import wpilib
import wpilib.simulation

import wpimath.system.plant
from pyfrc.physics.core import PhysicsInterface

import math
import typing

if typing.TYPE_CHECKING:
    from robot import MyRobot


class PhysicsEngine:
    """
    Simulates an arm
    """

    def __init__(self, physics_controller: PhysicsInterface, robot: "MyRobot"):
        """
        :param physics_controller: `pyfrc.physics.core.Physics` object
                                   to communicate simulation effects to
        :param robot: your robot object
        """

        self.physics_controller = physics_controller

        # The arm gearbox represents a gearbox containing two Vex 775pro
        # motors.
        self.arm_gearbox = wpimath.system.plant.DCMotor.vex775Pro(1)

        # Simulation classes help us simulate what's going on, including gravity.
        # This sim represents an arm with 1 775s, a 600:1 reduction, a mass of 5kg,
        # 30in overall arm length, range of motion in [-75, 255] degrees, and noise
        # with a standard deviation of 1 encoder tick.
        self.arm_sim = wpilib.simulation.SingleJointedArmSim(
            self.arm_gearbox,
            1200.0,
            wpilib.simulation.SingleJointedArmSim.estimateMOI(0.762, 5),
            0.762,
            math.radians(-90),
            math.radians(255),
            5,
            True,
            [robot.kArmEncoderDistPerPulse],
        )
        self.encoder_sim = wpilib.simulation.EncoderSim(robot.encoder)
        self.motor_sim = wpilib.simulation.PWMSim(robot.motor.getChannel())

        # Create a Mechanism2d display of an Arm
        self.mech2d = wpilib.Mechanism2d(60, 60)
        self.arm_base = self.mech2d.getRoot("ArmBase", 30, 30)
        self.arm_tower = self.arm_base.appendLigament(
            "Arm Tower", 30, -90, 6, wpilib.Color8Bit(wpilib.Color.kBlue)
        )
        self.arm = self.arm_base.appendLigament(
            "Arm", 30, self.arm_sim.getAngle(), 6, wpilib.Color8Bit(
                wpilib.Color.kYellow))

        # Put Mechanism to SmartDashboard
        wpilib.SmartDashboard.putData("Arm Sim", self.mech2d)

    def update_sim(self, now: float, tm_diff: float) -> None:
        """
        Called when the simulation parameters for the program need to be
        updated.

        :param now: The current time as a float
        :param tm_diff: The amount of time that has passed since the last
                        time that this function was called
        """

        # First, we set our "inputs" (voltages)
        self.arm_sim.setInput(
            0,
            self.motor_sim.getSpeed() *
            wpilib.RobotController.getInputVoltage())

        # Next, we update it
        self.arm_sim.update(tm_diff)

        # Finally, we set our simulated encoder's readings and simulated battery
        # voltage
        self.encoder_sim.setDistance(self.arm_sim.getAngle())
        # SimBattery estimates loaded battery voltage
        # wpilib.simulation.RoboRioSim.setVInVoltage(
        #     wpilib.simulation.BatterySim
        # )

        # Update the mechanism arm angle based on the simulated arm angle
        # -> setAngle takes degrees, getAngle returns radians... >_>
        self.arm.setAngle(math.degrees(self.arm_sim.getAngle()))
