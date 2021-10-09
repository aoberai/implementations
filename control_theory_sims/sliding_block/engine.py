import time
import pygame
import constants as ct

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
        self.init_time = time.time()
        self.display_dims = display_dims
    def step(self, current_time, applied_f):
        normal_f = self.g*self.mass
        fric_f = normal_f * self.k_coef_fric
        net_f = (applied_f - fric_f)
        a = net_f/self.mass
        d_time = current_time - self.init_time
        self.x_h = self.x_init[0] + self.v_init[0]*d_time + 0.5*a*d_time**2
        self.v_h = self.v_init[0] + a*d_time
        return (self.x_h, self.v_h, net_f)

class _Surface:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pass
    def step(self):
        pass

class Components:
    def __init__(self, mass, surface_width, surface_height, x_init, v_init, s_coef_fric, k_coef_fric, block_display_dims, wants_render=True, screen_dims=(5, 5)):
        '''
        All units, including display_dims, is in meters
        '''
        self._block = _Block(mass, x_init, v_init, s_coef_fric, k_coef_fric, display_dims=block_display_dims)
        self._surface = _Surface(surface_width, surface_height)
        self.screen_dims = screen_dims
        if wants_render:
            pygame.init()
            self.pyg_surface = pygame.display.set_mode((int(screen_dims[0]*ct.PIXELS_PER_METER), int(screen_dims[1]*ct.PIXELS_PER_METER)))
    def step(self, current_time, applied_f):
        self._surface.step()
        # TODO: add contact param which checks for contact from all sides
        block_state = self._block.step(current_time, applied_f)
        return block_state
    def render(self, block_state):
        self.pyg_surface.fill((0, 0, 0))
        # TODO: put pygame rect in class
        surf_rect = pygame.Rect(0, int(ct.PIXELS_PER_METER*(self.screen_dims[1]-self._surface.height)), int(self._surface.width*self.screen_dims[0]*ct.PIXELS_PER_METER**2), int(self._surface.height*self.screen_dims[1]*ct.PIXELS_PER_METER**2))
        block_rect = pygame.Rect(int(ct.PIXELS_PER_METER*block_state[0]), int(ct.PIXELS_PER_METER*(self.screen_dims[1] - self._surface.height - self._block.display_dims[1])), int(ct.PIXELS_PER_METER*self._block.display_dims[0]), int(ct.PIXELS_PER_METER*self._block.display_dims[1]))
        pygame.draw.rect(self.pyg_surface, ct.GREY, surf_rect)
        pygame.draw.rect(self.pyg_surface, ct.WHITE, block_rect)
        pygame.display.flip()



