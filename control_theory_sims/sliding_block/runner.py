from engine import Components
import time
import math

# Implement y component forces and interactions with static obj
components = Components(mass=10, surface_width=0.3, surface_height=0.01, x_init=(0.1, 0), v_init=(0.01, 0), s_coef_fric=0.3, k_coef_fric=0.15, block_display_dims=(0.01, 0.01), wants_render=True, screen_dims=(0.2, 0.2))

def force_func(time):
    return 5*(math.sin(time)+0.5)

while True:
    curr_time = time.time()
    x_h, v_h, f_h = components.step(curr_time, applied_f=force_func(curr_time))
    components.render(block_state=(x_h, v_h,))
    print("Position: %0.3f; Velocity: %0.3f, Net Horizontal Force: %0.3f" % (x_h, v_h, f_h))
    time.sleep(0.1)


