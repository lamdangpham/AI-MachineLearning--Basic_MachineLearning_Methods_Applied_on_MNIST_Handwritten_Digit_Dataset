import numpy as np
import os

class dnn_bl01_para(object):
    """
    define a class to store parameters
    """

    def __init__(self):

        #=======Layer 01: Final layer
        self.l01_fc             = 10   
        self.l01_is_act         = False
        self.l01_act_func       = 'RELU'
        self.l01_is_drop        = False
        self.l01_drop_prob      = 1

