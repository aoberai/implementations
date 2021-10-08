from engine import Components
import time

# Implement y component forces and interactions with static obj
components = Components(mass=10, surface_width=1, surface_height=0.1, x_init=(2, 0), v_init=(10, 0), s_coef_fric=0.3, k_coef_fric=0.15, block_display_dims=(50, 50), wants_render=True)

while True:
    curr_time = time.time()
    x_h, v_h, f_h = components.step(curr_time, applied_f=0.1)
    components.render(block_state=(x_h, v_h,))
    print("Position: %0.3f; Velocity: %0.3f, Net Horizontal Force: %0.3f" % (x_h, v_h, f_h))
    time.sleep(0.1)


