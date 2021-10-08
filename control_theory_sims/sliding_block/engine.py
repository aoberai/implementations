import time
import pygame

class _Block:
    g = 9.8
    def __init__(self, mass, x_init, v_init, s_coef_fric, k_coef_fric, display_dims):
        self.mass = mass
        self.x_init = x_init[0]
        self.v_init = v_init[0]
        self.s_coef_fric = s_coef_fric
        self.k_coef_fric = k_coef_fric
        self.x_h = self.x_init
        self.v_h = self.v_init 
        self.init_time = time.time()
        self.display_dims = display_dims
        pass
    def step(self, current_time, applied_f):
        normal_f = self.g*self.mass
        fric_f = normal_f * self.k_coef_fric
        net_f = (applied_f - fric_f)
        a = net_f/self.mass
        d_time = current_time - self.init_time
        self.x_h = self.x_init + self.v_init*d_time + 0.5*a*d_time**2
        self.v_h = self.v_init + a*d_time
        return (self.x_h, self.v_h, net_f)

class _Surface:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pass
    def step(self):
        pass

class Components:
    def __init__(self, mass, surface_width, surface_height, x_init, v_init, s_coef_fric, k_coef_fric, block_display_dims, wants_render=True):
        self._block = _Block(mass, x_init, v_init, s_coef_fric, k_coef_fric, display_dims=block_display_dims)
        self._surface = _Surface(surface_width, surface_height)
        if wants_render:
            self._GREY   = (210, 210 ,210)
            self._WHITE  = (255, 255, 255)
            self._SCREEN_WIDTH = 1000
            self._SCREEN_HEIGHT = 700
            pygame.init()
            self.pyg_surface = pygame.display.set_mode((self._SCREEN_WIDTH, self._SCREEN_HEIGHT,))
            
    def step(self, current_time, applied_f):
        self._surface.step()
        block_state = self._block.step(current_time, applied_f)
        return block_state
    def render(self, block_state):
        self.pyg_surface.fill((0, 0, 0))
        pygame.draw.rect(self.pyg_surface, self._GREY, pygame.Rect(0, self._SCREEN_HEIGHT - self._SCREEN_HEIGHT * self._surface.height, self._surface.width * self._SCREEN_WIDTH, self._surface.height * self._SCREEN_HEIGHT))
        pygame.draw.rect(self.pyg_surface, self._WHITE, pygame.Rect((block_state[0], self._SCREEN_HEIGHT - self._SCREEN_HEIGHT * self._surface.height - self._block.display_dims[1]), (self._block.display_dims[0], self._block.display_dims[1])))
        pygame.display.flip()



