import numpy as np
from scipy.stats import truncnorm
import random
from conf import logistic_map

class Worker:
    def __init__(self, id=-1, B=-1, T=-1, n=-1, K=-1, mean=-1, cost=1, variance=-1, bundle=[], x=-1, y=-1, payment=-1,
                 privacy_budget=0.5, weight=1, curve_size=-1, bid=-1, q=-1):
        self.id=id
        self.B=B
        self.n=n
        self.T=T
        self.K=K
        self.mean=mean
        self.cost=cost
        self.variance=variance
        self.bundle=bundle
        self.e_list = [0 for i in range(self.T)]
        self.x=x
        self.y=y
        self.task_sequence = []
        self.payment=payment
        self.other_payment=0
        self.reward_list=[]
        self.reward_noise_list = []
        self.utility_list = []
        self.privacy_budget=privacy_budget
        self.weight=weight
        self.weight_2=-1
        self.probability=-1
        self.probability_2=-1
        self.measurement=-1
        self.weight_list=[]
        self.probability_list=[]
        self.curver_size=curve_size

        self.payload = -1
        self.optimal_payload = -1
        self.utility = -1
        self.optimal_reward_unit = -1
        self.bid = bid
        self.Q = (-1, -1)
        self.payment_list = {}
        self.mu_q = q
        self.sigma_q = random.uniform(0, min(q / 3, (1 - q) / 3))

    def calUtility_drone(self):
        #initial the quality
        if self.curver_size == 0:
            self.curver_size = 0.01
        if self.curver_size == 1.0:
            self.curver_size = 0.99
        self.online_utility_list = [i for i in truncnorm.rvs(a = -self.curver_size / self.variance, b = self.curver_size / self.variance,
                                                                loc=self.curver_size, scale=self.variance, size=self.T + 1)]
        self.reward_list.append(self.online_utility_list[0])  #each worker is assigned at least once initially
        reward_noise = (self.online_utility_list[0] + np.random.laplace(loc=0, scale=1 / self.privacy_budget, size=1)[0])
        while (reward_noise < 0):
            reward_noise = (self.online_utility_list[0] + np.random.laplace(loc=0, scale=1 / self.privacy_budget, size=1)[0])
        self.reward_noise_list.append(reward_noise)  # each worker is assigned at least once initially

    def SamplePayload(self):
        # 生成n个随机样本
        U = np.random.uniform(0, 1, self.T+1)
        # 根据CDF的反函数生成样本
        self.online_payload_list = [i for i in 5 * U ** (4/5)]
        self.payload = self.online_payload_list[0]

    def draw(self):
        """ universally used in algorithm 1 & 2 """
        return random.gauss(self.mu_q, self.sigma_q)

    def __str__(self):
        return 'Worker(id,payload,reward):'+'({},{},{})'.format(self.id,self.payload,self.reward_list)

class Task:
    def __init__(self, id=-1, x=-1, y=-1, worker_id=-1):
        self.id=id
        self.x=x
        self.y=y
        self.worker_id=worker_id

if __name__ == '__main__':
    print("Define worker class and task class.")
