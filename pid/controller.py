import time

class PIDController:
    def __init__(self, Kp=0, Ki=0, Kd=0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral_error = 0
        self.first_run = True
        self.last_time = time.time()
    def feedback(self, error):
        # TODO: involve time
        if self.first_run:
            self.last_error = error
            self.first_run = not self.first_run
        self.integral_error += error
        dt = (time.time() - self.last_time)
        output =  error * self.Kp + ((error-self.last_error) * -self.Kd)/dt + self.integral_error * self.Ki
        self.last_error = error
        self.last_time = time.time()
        return output

