import numpy as np
import random
from utils.options import args_parser
args=args_parser()

class user:
    def __init__(self):
        self.save=random.randint(80000,100000)
        self.personal_b=(3-random.random())/3
        self.personal_a = (3 - random.random()) / 3
        self.ask=[random.uniform(1,2) for i in range(args.num_users)]
        self.w=None
        self.loss=None
    def buy(self,sim):
        return (1 - 1 / (1 + np.exp(-(sim - 0.4) * 10))) * 50 *self.personal_b
    def set_save(self,a):
        self.save-=a




