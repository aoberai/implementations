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
    def __init__(self, mass, surface_width, surface_height, x_init, v_init, s_coef_fric, k_coef_fric, block_display_dims, pid_init, wants_render=True, screen_dims=(5, 5)):
        '''
        All units, including display_dims, is in meters
        '''
        self._block = _Block(mass, x_init, v_init, s_coef_fric, k_coef_fric, display_dims=block_display_dims)
        self._surface = _Surface(surface_width, surface_height)
        self.screen_dims = screen_dims
        self.reference_rect_dims = (0.002, 0.01)
        self.ref_save_counter = 0
        if wants_render:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
            pygame.init()
            # pygame.key.set_repeat(10, 500)
            self.font = pygame.font.Font(None, 32)
            self.kP_text = 'kP:' + str(pid_init["kP"])
            self.kI_text = 'kI:' + str(pid_init["kI"])
            self.kD_text = 'kD:' + str(pid_init["kD"])
            self.kP_rect = pygame.Rect(100, 100, 140, 32)
            self.kI_rect = pygame.Rect(250, 100, 140, 32)
            self.kD_rect = pygame.Rect(400, 100, 140, 32)
            self.save_text = "Save"
            self.pid_save_rect = pygame.Rect(550, 100, 80, 32)
            self.active_textbox = None
            self.pyg_surface = pygame.display.set_mode((int(screen_dims[0]*ct.PIXELS_PER_METER), int(screen_dims[1]*ct.PIXELS_PER_METER)))
    def step(self, current_time, applied_f):
        self._surface.step()
        # TODO: add contact param which checks for contact from all sides
        block_state = self._block.step(current_time, applied_f)
        return block_state
    def render(self, block_state, reference_x):
        events = pygame.event.get()
        if (pygame.mouse.get_pressed()[2]):
            self.ref_save_counter+=1
        if self.ref_save_counter == 10:
            self.ref_save_counter = 0
            reference_x[0] = pygame.mouse.get_pos()[0]/ct.PIXELS_PER_METER

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.kP_rect.collidepoint(event.pos):
                    self.active_textbox = "kP_textbox"
                if self.kI_rect.collidepoint(event.pos):
                    self.active_textbox = "kI_textbox"
                if self.kD_rect.collidepoint(event.pos):
                    self.active_textbox = "kD_textbox"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    if self.kP_text[-1] != ":" and self.active_textbox == "kP_textbox":
                        self.kP_text = self.kP_text[:-1]
                    if self.kI_text[-1] != ":" and self.active_textbox == "kI_textbox":
                        self.kI_text = self.kI_text[:-1]
                    if self.kD_text[-1] != ":" and self.active_textbox == "kD_textbox":
                        self.kD_text = self.kD_text[:-1]
                else:
                    if len(self.kP_text) <= 7 and self.active_textbox == "kP_textbox":
                        self.kP_text += event.unicode 
                    if len(self.kI_text) <= 7 and self.active_textbox == "kI_textbox":
                        self.kI_text += event.unicode
                    if len(self.kD_text) <= 7 and self.active_textbox == "kD_textbox":
                        self.kD_text += event.unicode
                if event.key == pygame.K_LEFT:
                    reference_x[0] -= 0.005
                if event.key == pygame.K_RIGHT:
                    reference_x[0] += 0.005
        self.pyg_surface.fill(ct.BLACK)
        pygame.draw.rect(self.pyg_surface, ct.WHITE, self.kP_rect)
        kP_text_surface = self.font.render(self.kP_text, True, ct.BLACK)
        pygame.draw.rect(self.pyg_surface, ct.WHITE, self.kI_rect)
        kI_text_surface = self.font.render(self.kI_text, True, ct.BLACK)
        pygame.draw.rect(self.pyg_surface, ct.WHITE, self.kD_rect)
        kD_text_surface = self.font.render(self.kD_text, True, ct.BLACK)
        pygame.draw.rect(self.pyg_surface, ct.WHITE, self.pid_save_rect)
        save_text_surface = self.font.render(self.save_text, True, ct.BLACK)
        self.pyg_surface.blit(kP_text_surface, (self.kP_rect.x+5, self.kP_rect.y+5))
        self.pyg_surface.blit(kI_text_surface, (self.kI_rect.x+5, self.kI_rect.y+5))
        self.pyg_surface.blit(kD_text_surface, (self.kD_rect.x+5, self.kD_rect.y+5))
        self.pyg_surface.blit(save_text_surface, (self.pid_save_rect.x+5, self.pid_save_rect.y+5))
        self.kP_rect.w = kP_text_surface.get_width()+10
        self.kI_rect.w = kI_text_surface.get_width()+10
        self.kD_rect.w = kD_text_surface.get_width()+10
        surf_rect = pygame.Rect(0, int(ct.PIXELS_PER_METER*(self.screen_dims[1]-self._surface.height)), int(self._surface.width*self.screen_dims[0]*ct.PIXELS_PER_METER**2), int(self._surface.height*self.screen_dims[1]*ct.PIXELS_PER_METER**2))
        block_rect = pygame.Rect(int(ct.PIXELS_PER_METER*block_state[0]), int(ct.PIXELS_PER_METER*(self.screen_dims[1] - self._surface.height - self._block.display_dims[1])), int(ct.PIXELS_PER_METER*self._block.display_dims[0]), int(ct.PIXELS_PER_METER*self._block.display_dims[1]))
        reference_rect = pygame.Rect(int(ct.PIXELS_PER_METER*(reference_x[0]+self._block.display_dims[0]/2 - self.reference_rect_dims[0]/2)), int(ct.PIXELS_PER_METER*(self.screen_dims[1] - self._surface.height - self.reference_rect_dims[1])), int(ct.PIXELS_PER_METER*self.reference_rect_dims[0]), int(ct.PIXELS_PER_METER*self.reference_rect_dims[1]))
        pygame.draw.rect(self.pyg_surface, ct.GREY, surf_rect)
        pygame.draw.rect(self.pyg_surface, ct.WHITE, block_rect)
        pygame.draw.rect(self.pyg_surface, ct.RED, reference_rect)
        pygame.display.flip()

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.pid_save_rect.collidepoint(event.pos):
                    try:
                        return {"kP":float(self.kP_text[3:]), "kI":float(self.kI_text[3:]), "kD":float(self.kD_text[3:])}
                    except Exception as e:
                        return None
        return None



