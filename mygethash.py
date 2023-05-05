import random

maxval = 10


def gethash(primer, N):
    a = random.randint(1, maxval)
    b = random.randint(0, maxval)

    def hash(x):
        return ((a * x + b) % primer) % N

    return hash
