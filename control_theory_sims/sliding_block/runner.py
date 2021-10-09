from simulation import Simulation
import time

# Implement y component forces and interactions with static obj
simulation = Simulation(mass=1, surface_width=0.5, surface_height=0.01, x_init=(0, 0), v_init=(0.3, 0), s_coef_fric=0.03, k_coef_fric=0.015, block_display_dims=(0.01, 0.01), wants_render=True, screen_dims=(0.5, 0.2))

reference_x = (0.1, 0)

applied_f = 0
x_h, v_h, f_h = 0, 0, 0

while True:
    curr_time = time.time()
    x_h, v_h, f_h = simulation.step(curr_time, applied_f=applied_f)
    error = (reference_x[0] - x_h)
    simulation.render(block_state=(x_h, v_h,), reference_x=reference_x)
    print("Error: %0.5f; Applied Force %0.5f; Position: %0.3f; Velocity: %0.3f, Net Horizontal Force: %0.3f" % (error, applied_f, x_h, v_h, f_h))
    # input()


