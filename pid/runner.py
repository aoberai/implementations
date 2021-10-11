from simulation import Simulation
from controller import PIDController
import time

PID = {"kP":70, "kI":40, "kD":10} # Implement IZone in display
controller = PIDController(Kp=PID["kP"], Ki=PID["kI"], Kd=PID["kD"], IZone=0.15)
simulation = Simulation(mass=1, surface_width=0.5, surface_height=0.01, x_init=(0, 0), v_init=(0.03, 0), s_coef_fric=0.03, k_coef_fric=0.015, block_display_dims=(0.01, 0.01), pid_init=PID, wants_render=True, screen_dims=(0.5, 0.2))

reference_x = [0.1, 0] # Press left and right arrow keys to move reference
                       # Right click to move reference position to under cursor

applied_f = 0
error = 0
x_h, v_h, f_h = 0, 0, 0

while True:
    curr_time = time.time()
    control_input = controller.feedback(error)
    x_h, v_h, f_h = simulation.step(curr_time, applied_f=control_input)
    error = (reference_x[0] - x_h)
    temp_PID = simulation.render(block_state=(x_h, v_h,), reference_x=reference_x)
    if temp_PID != None:
        PID = temp_PID
    controller.update_gains(Kp=PID["kP"], Ki=PID["kI"], Kd=PID["kD"])
    print("PID Gains: " + controller.to_string() + "; Control Input: %0.5f; Error: %0.5f; Applied Force %0.5f; Position: %0.3f; Velocity: %0.3f, Net Horizontal Force: %0.3f" % (control_input, error, applied_f, x_h, v_h, f_h))
    # input()


