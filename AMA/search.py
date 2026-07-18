import numpy as np


class AdaptiveLocalSearch:
    def __init__(self, time_matrix, delta=0.1):
        self.time = time_matrix

        # initial probabilities
        self.rho_SI = 0.5
        self.rho_MI = 0.5
        self.delta = delta

        # eta accumulators
        self.sum_eta_SI = 0.0
        self.count_SI = 0
        self.sum_eta_MI = 0.0
        self.count_MI = 0

        self.history = []
        self.generation_history =  []
        self.neighbor_history = []

    def select_operator(self):
        r = np.random.rand()
        return ("SI", r) if r < self.rho_SI else ("MI", r)

    def compute_delta(self, tour, move_i, move_j):
        """
        Delta = (ac + bd) - (ab + cd)
        """
        a = 0 if move_i == 0 else tour[move_i-1]
        b = tour[move_i]
        c = tour[move_j]
        d = 0 if move_j == len(tour)-1 else tour[move_j+1]

        ab = self.time[a][b]
        cd = self.time[c][d]
        ac = self.time[a][c]
        bd = self.time[b][d]

        delta = (ac + bd) - (ab + cd)

        edge_info = {
            "tour_before": tour.tolist(),
            "move_i": int(move_i),
            "move_j": int(move_j),
            "a": int(a),
            "b": int(b),
            "c": int(c),
            "d": int(d),
            "ab": float(ab),
            "cd": float(cd),
            "ac": float(ac),
            "bd": float(bd)
        }

        return delta, edge_info

    def single_inversion(self, tour):
        """
        SI: inversion dalam satu segmen
        """
        if len(tour) < 2:
            return tour.copy(), float("inf"), None

        move_i, move_j = sorted(np.random.choice(len(tour), 2, replace=False))

        delta, edge_info = self.compute_delta(tour, move_i, move_j)

        new_tour = tour.copy()
        new_tour[move_i:move_j+1] = new_tour[move_i:move_j+1][::-1]

        edge_info["tour_after"] = new_tour.tolist()

        return new_tour, delta, edge_info

    def multiple_inversion(self, tour):
        """
        MI: inversion lebih panjang / wider move
        """
        if len(tour) < 3:
            return tour.copy(), float("inf"), {
                "tour_before": tour.tolist(),
                "tour_after": tour.tolist(),
                "move_i": None,
                "move_j": None,
                "a": None,
                "b": None,
                "c": None,
                "d": None,
                "ab": None,
                "ac": None,
                "bd": None,
                "cd": None,
            }

        move_i = np.random.randint(0, len(tour)-2)
        move_j = np.random.randint(move_i+2, len(tour))

        delta, edge_info = self.compute_delta(tour, move_i, move_j)

        new_tour = tour.copy()
        new_tour[move_i:move_j+1] = new_tour[move_i:move_j+1][::-1]

        edge_info["tour_after"] = new_tour.tolist()

        return new_tour, delta, edge_info

    def compute_eta(self, f_old, f_new):
        if f_old <= 0:
            return 0.0
        return (f_old - f_new) / f_old

    def record_eta(self, operator_type, eta):
        # print("RECORDING", operator_type, eta)

        if operator_type == "SI":
            self.sum_eta_SI += eta
            self.count_SI += 1
        elif operator_type == "MI":
            self.sum_eta_MI += eta
            self.count_MI += 1

    def update_probabilities_per_generation(self, generation):
        """ print(
            "BEFORE UPDATE |",
            "sum_eta_SI:", self.sum_eta_SI,
            "count_SI:", self.count_SI,
            "sum_eta_MI:", self.sum_eta_MI,
            "count_MI:", self.count_MI
        )"""

        avg_eta_SI = self.sum_eta_SI / self.count_SI if self.count_SI > 0 else 0
        avg_eta_MI = self.sum_eta_MI / self.count_MI if self.count_MI > 0 else 0

        #print("AVG SI:", avg_eta_SI, "AVG MI:", avg_eta_MI)

        self.rho_SI += self.delta * avg_eta_SI
        self.rho_MI += self.delta * avg_eta_MI

        total = self.rho_SI + self.rho_MI
        self.rho_SI /= total
        self.rho_MI /= total

        # print("UPDATED RHO:", self.rho_SI, self.rho_MI)

        self.generation_history.append({
            "generation": generation,
            "avg_eta_SI": avg_eta_SI,
            "avg_eta_MI": avg_eta_MI,
            "rho_SI": self.rho_SI,
            "rho_MI": self.rho_MI,
            "sum_eta_SI": self.sum_eta_SI,
            "count_SI": self.count_SI,
            "sum_eta_MI": self.sum_eta_MI,
            "count_MI": self.count_MI
        })

        # reset
        self.sum_eta_SI = 0.0
        self.count_SI = 0
        self.sum_eta_MI = 0.0
        self.count_MI = 0
