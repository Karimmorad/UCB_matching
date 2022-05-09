from UCB_matching import UCB_Matching
import numpy as np
from scipy.optimize import linear_sum_assignment
from CUSUM import CUSUM


class CUSUM_UCB_Matching(UCB_Matching):
    def __init__(self, n_arms, n_rows, n_cols, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms, n_rows, n_cols)
        self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha = alpha

    def pull_arm(self):
        if np.random.binomial(1, 1-self.alpha):
            upper_conf = self.empirical_means + self.confidence
            upper_conf[np.isinf(upper_conf)] = 1e3
            row_ind, col_ind = linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
            return row_ind ,col_ind
        else:
            costs_random = np.random.randint(0, 10, size=(self.n_rows, self.n_cols))
            return linear_sum_assignment(costs_random)

    def update(self, pulled_arm, reward):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arm, (self.n_rows, self.n_cols))
        for pulled_arm, reward in zip(pulled_arm_flat, reward):
            if self.change_detection[pulled_arm].update(reward):
                self.detections[pulled_arm].append(reward)
                self.valid_rewards_per_arms[pulled_arm] = []
                self.change_detection[pulled_arm].reset()
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arms])
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(total_valid_samples) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)


if __name__ == '__main__':
    p0 = np.array([[1/4, 1, 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
    p1 = np.array([[1, 1/4, 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
    p2 = np.array([[1, 1/4, 1/4], [1/2, 1, 1/4], [1/4, 1/4, 1]])
    p = [p0, p1, p2]
    T = 3000
    n_exp = 10
    regret_cusum = np.zeros(n_exp, T)
    regret_ucb = np.zeros(n_exp, T)
    detections = [[] for _ in range(n_exp)]
    M = 100
    eps = 0.1
    h = np.log(T)*2
    for j in range(n_exp):
        e_UCB = No
