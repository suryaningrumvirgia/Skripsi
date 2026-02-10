import numpy as np
from local_search import lso, LSO_2OPT

class AdaptiveLocalSearch:
    def __init__(self, delta=0.1, mi_max_iter=50):
        # probabilitas awal
        self.rho_SI = 0.5
        self.rho_MI = 0.5
        self.delta = delta

        # intensitas MI
        self.mi_max_iter = mi_max_iter

    def select_operator(self):
        """
        Pilih SI atau MI berdasarkan probabilitas
        """
        if np.random.rand() < self.rho_SI:
            return "SI", 1                  # single inverse
        else:
            return "MI", self.mi_max_iter  # multiple inverse

    def compute_eta(self, gain, fitness_ini):
        """
        η = (f_imp - f_ini) / f_ini
        gain < 0 berarti improvement
        """
        if gain >= 0 or fitness_ini <= 0:
            return 0.0
        return (-gain) / fitness_ini

    def update_probabilities(self, eta_SI, eta_MI):
        """
        Persamaan (2.12)–(2.15)
        """
        self.rho_SI += self.delta * eta_SI
        self.rho_MI += self.delta * eta_MI

        total = self.rho_SI + self.rho_MI
        if total > 0:
            self.rho_SI /= total
            self.rho_MI = 1.0 - self.rho_SI
