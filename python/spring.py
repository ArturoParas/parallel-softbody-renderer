
class Spring:

    damping_constant = 0.999

    def __init__(self, circle1, circle2, rest_length, k, tag=""):
        self.circle1 = circle1
        self.circle2 = circle2
        self.rest_length = rest_length
        self.k = k

        self.tag = tag
