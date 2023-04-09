import time

class Timer:
    def __init__(self):
        self.time = 0

    def start(self):
        self.time = time.time()

    def end(self, message = ""):
        t = time.time() - self.time
        print("%s: %.2f" %(message, t))