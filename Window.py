import math


class Window:
    def __getitem__(self, item):
        self.item = item

    def __init__(self, windowSize):
        self.n = windowSize

        # Make a Hamming window
        self.window = [0.0]*self.n
        for i in range(0, self.n):
            self.window[i] = 0.54 - 0.46 * math.cos(2 * math.pi * i / (self.n - 1))

    def applyWindow(self, buffer):
        for i in range(0, self.n):
            buffer[i] *= self.window[i]
