import numpy as np
from typing import Dict, Any


class DataGenBinaryClassifierScores:
    def __init__(self, class_params: Dict[Any, Any], c0_sample_size: int =200,
                 c1_sample_size: int =200):
        '''
        :param class_params: alpha, beta (beta distribution parameters) for each class
        :param c0_sample_size: the sample can be imbalanced in c0 and c1 classes
        :param c1_sample_size:
        '''
        self.class0_alpha = class_params.get('class0_alpha', 2)
        self.class0_beta = class_params.get('class0_beta', 2)
        self.class1_alpha = class_params.get('class1_alpha', 1)
        self.class1_beta = class_params.get('class1_beta', 1)
        self.c0_sample_size = c0_sample_size
        self.c1_sample_size = c1_sample_size
        self.scores = None

    def generate_samples(self):
        class_0 = np.random.beta(self.class0_alpha, self.class0_beta, size=self.c0_sample_size)
        class_1 = np.random.beta(self.class1_alpha, self.class1_beta, size=self.c1_sample_size)
        self.scores = {'class_0': class_0, 'class_1': class_1}
        return self.scores
