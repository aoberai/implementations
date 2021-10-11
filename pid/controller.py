import time

class PIDController:
    def __init__(self, Kp=0, Ki=0, Kd=0, IZone=None, error_clamp=None, output_clamp=None):
        self.Kp, self.Ki, self.Kd, self.IZone = Kp, Ki, Kd, IZone
        self.integral_error = 0
        self.first_run = True
        self.last_time = time.time()
        self.error_clamp = error_clamp
        self.output_clamp = output_clamp
    def update_gains(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
    def feedback(self, error):
        if self.error_clamp != None:
            error = max(min(self.error_clamp[1], error), self.error_clamp[0])
        if self.first_run:
            self.last_error = error
            self.first_run = not self.first_run

        dt = (time.time() - self.last_time)
        self.integral_error += -self.integral_error if abs(error) > self.IZone else error * dt
        output =  error * self.Kp + ((error-self.last_error) * self.Kd)/dt +  self.integral_error * self.Ki
        self.last_error = error
        self.last_time = time.time()
        if self.output_clamp != None:
            output = max(min(self.output_clamp[1], output), self.output_clamp[0])
        return output
    def zero_integrator(self):
        self.integral_error = 0
    def to_string(self):
        return "kP: %0.4f; kI: %0.4f; kD: %0.4f; IZone: %0.4f " % (self.Kp, self.Ki, self.Kd, self.IZone)

