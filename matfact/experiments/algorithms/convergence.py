from collections import defaultdict

import numpy as np 
import pandas as pd 


def convergence_monitor(M, error_tol=1e-4):
    """Track convergence of the matrix completion process by measuring the
    difference between consecutive estimates.
    """
    return MonitorFactorUpdate(M=M, tol=error_tol)


class MonitorFactorUpdate:

    def __init__(self, M, tol=1e-6):

        self.M = M
        self.tol = tol 

        self.n_iter_ = 0
        self.update_ = []
        self.convergence_rate_ = []
    
    def _should_stop(self, M_new):
        
        update = float(np.linalg.norm(M_new - self.M) ** 2 / np.linalg.norm(self.M) ** 2)

        if np.isnan(update):
            raise ValueError("Update value is NaN")

        self.update_.append(update)
            
        return update < self.tol

    def track_convergence_rate(self, M_new):
        
        self.Mpp = self.Mp
        self.Mp = self.M
        self.M = M_new

        a = np.linalg.norm(self.M - self.Mp)
        b = np.linalg.norm(self.Mp - self.Mpp) + 1e-12
        self.convergence_rate_.append(a / b)

    def converged(self, M_new):

        should_stop = self._should_stop(M_new)

        if self.n_iter_ > 0:
            self.track_convergence_rate(M_new=M_new)

        else:
            self.Mp = self.M
            self.M = M_new

        self.n_iter_ += 1

        return should_stop