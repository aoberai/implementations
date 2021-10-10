import time
import pygame
import constants as ct
import os

class _Block:
    g = 9.8
    def __init__(self, mass, x_init, v_init, s_coef_fric, k_coef_fric, display_dims):
        self.mass = mass
        self.x_init = x_init
        self.v_init = v_init
        self.s_coef_fric = s_coef_fric
        self.k_coef_fric = k_coef_fric
        self.x_h = self.x_init[0]
        self.v_h = self.v_init[0]
        self.last_time = time.time()
        self.display_dims = display_dims

    def step(self, current_time, applied_f):
        normal_f = self.g*self.mass
        kinetic2static_v = 0.01
        fric_f = ((normal_f * self.k_coef_fric) if abs(self.v_h) > kinetic2static_v else min(abs(applied_f), self.s_coef_fric * normal_f))
        fric_f *= -1 if self.v_h > 0 else 1
        net_f = (applied_f + fric_f)
        a = net_f/self.mass
        d_time = current_time - self.last_time
        self.x_h = self.x_h + self.v_h*d_time + 0.5*a*d_time**2
        self.v_h += a*d_time
        self.last_time = time.time()
        return (self.x_h, self.v_h, net_f)

class _Surface:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pass
    def step(self):
        pass

class Simulation:
    def __init__(self, mass, surface_width, surface_height, x_init, v_init, s_coef_fric, k_coef_fric, block_display_dims, wants_render=True, screen_dims=(5, 5)):
        '''
        All units, including display_dims, is in meters
        '''
        self._block = _Block(mass, x_init, v_init, s_coef_fric, k_coef_fric, display_dims=block_display_dims)
        self._surface = _Surface(surface_width, surface_height)
        self.screen_dims = screen_dims
        self.reference_rect_dims = (0.002, 0.01)
        if wants_render:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
            pygame.init()
            pygame.key.set_repeat(10, 500)
            self.pyg_surface = pygame.display.set_mode((int(screen_dims[0]*ct.PIXELS_PER_METER), int(screen_dims[1]*ct.PIXELS_PER_METER)))
    def step(self, current_time, applied_f):
        self._surface.step()
        # TODO: add contact param which checks for contact from all sides
        block_state = self._block.step(current_time, applied_f)
        return block_state
    def render(self, block_state, reference_x):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    reference_x[0] -= 0.005
                if event.key == pygame.K_RIGHT:
                    reference_x[0] += 0.005
        self.pyg_surface.fill((0, 0, 0))
        # TODO: put pygame rect in class
        surf_rect = pygame.Rect(0, int(ct.PIXELS_PER_METER*(self.screen_dims[1]-self._surface.height)), int(self._surface.width*self.screen_dims[0]*ct.PIXELS_PER_METER**2), int(self._surface.height*self.screen_dims[1]*ct.PIXELS_PER_METER**2))
        block_rect = pygame.Rect(int(ct.PIXELS_PER_METER*block_state[0]), int(ct.PIXELS_PER_METER*(self.screen_dims[1] - self._surface.height - self._block.display_dims[1])), int(ct.PIXELS_PER_METER*self._block.display_dims[0]), int(ct.PIXELS_PER_METER*self._block.display_dims[1]))
        reference_rect = pygame.Rect(int(ct.PIXELS_PER_METER*(reference_x[0]+self._block.display_dims[0]/2 - self.reference_rect_dims[0]/2)), int(ct.PIXELS_PER_METER*(self.screen_dims[1] - self._surface.height - self.reference_rect_dims[1])), int(ct.PIXELS_PER_METER*self.reference_rect_dims[0]), int(ct.PIXELS_PER_METER*self.reference_rect_dims[1]))
        pygame.draw.rect(self.pyg_surface, ct.GREY, surf_rect)
        pygame.draw.rect(self.pyg_surface, ct.WHITE, block_rect)
        pygame.draw.rect(self.pyg_surface, ct.RED, reference_rect)
        pygame.display.flip()



