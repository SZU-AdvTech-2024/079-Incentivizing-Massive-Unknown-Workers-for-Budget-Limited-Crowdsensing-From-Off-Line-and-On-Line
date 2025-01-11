import pandas as pd
from conf import *
import copy
import math
import heapq
from Worker import *
import matplotlib.pyplot as plt
import os

class Policy:
    def __init__(self):
        self.T = T
        self.n = n
        self.m = m
        self.B = B
        self.K = K
        self.privacy_budget = privacy_budget
        self.size = size
        self.count = count
        self.variance = variance
        self.worker_list = []
        self.task_list = []
        self.task_bundle_list = []
        self.b_max = b_max
        self.epsilon = epsilon

    # -------------------------------several assistant functions-----------------------------------
    def reset(self):
        self.T = T
        self.n = n
        self.m = m
        self.B = B
        self.K = K
        self.size = size
        self.count = count
        self.variance = variance

    # -------------------------------initialization functon-----------------------------------
    '''load workers from datasets'''
    def loadDrons(self):
        worker_list = []
        task_list = []
        #数据集下载链接：https://figshare.com/articles/dataset/Sawyer_Mill_Dam_Removal_Project_Upper_Impoundment_High_Altitude_Drone_Flight_Paths/14669481
        file = os.path.join(dataset_address, '2019-7-30_seds_angle1_100ft.csv')
        df = pd.read_csv(file)
        df['curvesize(ft)'] = (df['curvesize(ft)'] - df['curvesize(ft)'].min()) / (df['curvesize(ft)'].max() - df['curvesize(ft)'].min())
        for index, row in df.iterrows():
            worker = Worker(x=row["longitude"], y=row["latitude"], id=index, curve_size=row["curvesize(ft)"],
                            B=self.B, T=self.T, n=self.n, K=self.K, variance=self.variance, privacy_budget=self.privacy_budget)
            worker_list.append(worker)
            if len(worker_list) == self.n:
                break
        for worker in worker_list:
            worker.calUtility_drone()
        return worker_list, task_list

    # -------------------------------online policies-----------------------------------
    def online_policy_Greedy(self, worker_list, budget=-1, K=-1, Time=-1):
        temp_worker_list = copy.deepcopy(worker_list)
        utility_dict = {}
        t = 1
        remaining_budget = budget
        while remaining_budget >= 0:
            temp_worker_list=sorted(temp_worker_list, key=lambda x: x.reward_list[-1], reverse=True)
            temp_winner_list = temp_worker_list[:K]
            for winner in temp_winner_list:
                reward = winner.online_utility_list[t%Time]
                winner.reward_list.append(reward)
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            remaining_budget -= sum([i.reward_list[-1] for i in temp_winner_list])
            t = t + 1
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def online_policy_Random(self, worker_list, budget=-1, K=-1, Time=-1):
        temp_worker_list = copy.deepcopy(worker_list)
        utility_dict = {}
        t = 1
        remaining_budget = budget
        while remaining_budget >= 0:
            temp_winner_list = random.sample(temp_worker_list, K)
            for winner in temp_winner_list:
                reward = winner.online_utility_list[t%Time]
                winner.reward_list.append(reward)
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            remaining_budget -= sum([i.reward_list[-1] for i in temp_winner_list])
            t = t + 1
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def online_policy_AUCB(self, worker_list, budget=-1, K=-1, Time=-1):
        temp_worker_list = copy.deepcopy(worker_list)
        utility_dict = {}
        c_average=0.6
        t = 1
        remaining_budget = budget
        while remaining_budget >= 0:
            for i in temp_worker_list:
                average = np.mean(i.reward_list)
                a = math.sqrt((budget + 1) * math.log(t, math.e) / len(i.reward_list))/c_average
                e = a
                i.measurement=average+e
            temp_worker_list = sorted(temp_worker_list, key=lambda x: x.measurement, reverse=True)
            temp_winner_list = temp_worker_list[:K]
            for winner in temp_winner_list:
                reward = winner.online_utility_list[t%Time]
                winner.reward_list.append(reward)
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            remaining_budget -= sum([i.reward_list[-1] for i in temp_winner_list])
            t = t + 1
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def Baseline_CACI_OFF(self, worker_list, budget=-1, K=-1):
        temp_worker_list = copy.deepcopy(worker_list)
        for worker in temp_worker_list:
            worker.cost = random.uniform(0.1, 0.6)
            worker.bid = random.uniform(worker.cost, self.b_max)
        utility_dict = {}
        t = 1
        temp_worker_list = sorted(temp_worker_list, key=lambda x: x.reward_list[-1] / x.bid, reverse=True)
        temp_winner_list = temp_worker_list[:K]
        for worker in temp_winner_list:
            worker.other_payment = min(self.b_max, worker.reward_list[-1] / temp_worker_list[K].reward_list[-1] * temp_worker_list[K].bid)
        remaining_budget = budget
        while remaining_budget >= sum(worker.other_payment for worker in temp_winner_list):
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            remaining_budget -= sum(worker.other_payment for worker in temp_winner_list)
            t = t + 1
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def Baseline_CACI_ON(self, worker_list, budget=-1, K=-1):
        temp_worker_list = copy.deepcopy(worker_list)
        remaining_budget = budget
        utility_dict = {}
        t = 1
        for worker in temp_worker_list:
            worker.cost = random.uniform(0.1, 0.6)
            worker.bid = random.uniform(worker.cost, self.b_max)
        while True:
            temp_worker_list = sorted(temp_worker_list, key=lambda x: x.reward_list[-1] / x.bid, reverse=True)
            temp_winner_list = temp_worker_list[:K]
            for worker in temp_winner_list:
                worker.other_payment = min(self.b_max, worker.reward_list[-1] / temp_worker_list[K].reward_list[-1] * temp_worker_list[K].bid)
            if remaining_budget >= sum(worker.other_payment for worker in temp_winner_list):
                utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
                remaining_budget -= sum(worker.other_payment for worker in temp_winner_list)
                t = t + 1
            else:
                sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
                return sum_utility, utility_dict

    def online_policy_CACI_OFF(self, worker_list, budget=-1, K=-1, Time=-1):
        temp_worker_list = copy.deepcopy(worker_list)
        dimension = 2
        alpha_new = 1.5
        d = math.ceil(budget ** (1 / (3 * alpha_new + dimension)))
        context_space = np.linspace(0, 1, d)
        hypercubes = [(x, y) for x in context_space for y in context_space]
        lambda_counts = {}
        average_rewards = {}
        utility_dict = {}
        for worker in temp_worker_list:
            worker.cost = random.uniform(0.1, 0.6)
            worker.bid = random.uniform(worker.cost, self.b_max)
        miu_max = 6

        #Initialization:
        t = 0
        remaining_budget = budget
        for Q in hypercubes:
            lambda_counts[Q] = 0
            average_rewards[Q] = 0.0
        for worker in temp_worker_list:
            worker.Q = hypercubes[worker.id % len(hypercubes)]

        #Exploration phase:
        B_hash = ((self.b_max / miu_max **2)**(1/3)) * (d ** (dimension / 3)) * (budget ** (2/3) * np.log(budget))
        T = math.floor(B_hash / (K * self.b_max))
        for t in range(1, T+1):
            N_t = []
            for k in range(1, K + 1):
                l = ((t - 1) * K + k) % len(hypercubes)
                N_t.append(random.choice([worker for worker in temp_worker_list if worker.Q == hypercubes[l]]))
            for worker in temp_worker_list:
                if worker in N_t:
                    worker.other_payment = self.b_max
                    reward = worker.online_utility_list[t%Time]
                    worker.reward_list.append(reward)
                    if lambda_counts[worker.Q] > 0:
                        average_rewards[worker.Q] = (lambda_counts[worker.Q] * average_rewards[worker.Q] + reward) / lambda_counts[worker.Q]
                    else:
                        average_rewards[worker.Q] = 0
                    lambda_counts[worker.Q] += 1
                else:
                    worker.other_payment = 0
            utility_dict[t] = sum([i.reward_list[-1] for i in N_t])
            remaining_budget = remaining_budget - K * self.b_max
            if remaining_budget <= 0:
                break

        #Exploitation phase:
        t = t + 1
        for worker in temp_worker_list:
            miu_i = average_rewards[worker.Q] + np.sqrt(np.log(budget) / lambda_counts[worker.Q]) if lambda_counts[worker.Q] > 0 else 0
            worker.utility_list.append(miu_i)
        temp_worker_list = sorted(temp_worker_list, key=lambda x: x.utility_list[-1] / x.bid, reverse=True)
        temp_winner_list = temp_worker_list[:K]
        for worker in temp_winner_list:
            if temp_worker_list[K].utility_list[-1] == 0:
                worker.other_payment = 0
            else:
                worker.other_payment = min(self.b_max, worker.utility_list[-1] / temp_worker_list[K].utility_list[-1] * temp_worker_list[K].bid)
        while remaining_budget >= sum(worker.other_payment for worker in temp_winner_list):
            for worker in temp_worker_list:
                if worker in temp_winner_list:
                    reward = worker.online_utility_list[t%Time]
                    worker.reward_list.append(reward)
                    if lambda_counts[worker.Q] > 0:
                        average_rewards[worker.Q] = (lambda_counts[worker.Q] * average_rewards[worker.Q] + reward) / lambda_counts[worker.Q]
                    else:
                        average_rewards[worker.Q] = 0
                    lambda_counts[worker.Q] += 1
                else:
                    worker.other_payment = 0
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            remaining_budget = remaining_budget - sum(worker.other_payment for worker in temp_winner_list)
            t = t + 1
            for worker in temp_worker_list:
                miu_i = average_rewards[worker.Q] + np.sqrt(np.log(budget) / lambda_counts[worker.Q]) if lambda_counts[worker.Q] > 0 else 0
                worker.utility_list.append(miu_i)
            temp_worker_list = sorted(temp_worker_list, key=lambda x: x.utility_list[-1] / x.bid, reverse=True)
            temp_winner_list = temp_worker_list[:K]
            for worker in temp_winner_list:
                if temp_worker_list[K].utility_list[-1] == 0:
                    worker.other_payment = 0
                else:
                    worker.other_payment = min(self.b_max, worker.utility_list[-1] / temp_worker_list[K].utility_list[-1] * temp_worker_list[K].bid)
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def online_policy_CACI_ON(self, worker_list, budget=-1, K=-1, Time=-1):
        temp_worker_list = copy.deepcopy(worker_list)
        dimension = 2
        alpha_new = 1.5
        d = math.ceil(budget ** (1 / (3 * alpha_new + dimension)))
        context_space = np.linspace(0, 1, d)
        hypercubes = [(x, y) for x in context_space for y in context_space]
        lambda_counts = {}
        average_rewards = {}
        utility_dict = {}
        for worker in temp_worker_list:
            worker.cost = random.uniform(0.1, 0.6)
            worker.bid = random.uniform(worker.cost, self.b_max)

        #Initialization:
        t = 1
        remaining_budget = budget
        for worker in temp_worker_list:
            worker.Q = hypercubes[worker.id % len(hypercubes)]
        N_t = []
        for Q in hypercubes:
            worker = (random.choice([worker1 for worker1 in temp_worker_list if worker1.Q == Q]))
            worker.other_payment = self.b_max
            reward = worker.online_utility_list[t%Time]
            lambda_counts[Q] = 1
            average_rewards[Q] = reward
            N_t.append(worker)
        utility_dict[t] = sum([i.reward_list[-1] for i in N_t])
        remaining_budget = remaining_budget - len(hypercubes) * self.b_max

        #Exploration and Exploitation phase:
        t = t + 1
        while remaining_budget >= K * self.b_max:
            for worker in temp_worker_list:
                if lambda_counts[worker.Q] > 0:
                    worker.utility_list.append(average_rewards[Q] + np.sqrt((K+1) * np.log(t) / lambda_counts[Q]))
                else:
                    worker.utility_list.append(0)
            temp_worker_list = sorted(temp_worker_list, key=lambda x: x.utility_list[-1] / x.bid, reverse=True)
            temp_winner_list = temp_worker_list[:K]
            for worker in temp_winner_list:
                if temp_worker_list[K].utility_list[-1] == 0:
                    worker.other_payment = 0
                else:
                    worker.other_payment = min(self.b_max, worker.utility_list[-1] / temp_worker_list[K].utility_list[-1] * temp_worker_list[K].bid)
                    reward = worker.online_utility_list[t%Time]
                    worker.reward_list.append(reward)
                    if lambda_counts[worker.Q] > 0:
                        average_rewards[worker.Q] = (lambda_counts[worker.Q] * average_rewards[worker.Q] + reward) / lambda_counts[worker.Q]
                    else:
                        average_rewards[worker.Q] = 0
                    lambda_counts[worker.Q] += 1
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            remaining_budget = remaining_budget - sum(worker.other_payment for worker in temp_winner_list)
            t = t + 1
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def select_winners(self, worker_list, K, n_i, q_bar, w):
        selected_worker_list = []
        while len(selected_worker_list) < K:
            heap = []
            q_hat = q_bar + np.sqrt((K + 1) * np.log(np.sum(n_i)) / n_i)
            v = np.zeros(self.n)
            for worker in worker_list:
                v[worker.id] = np.maximum(q_hat[worker.id], v[worker.id])
            base = np.dot(w, v)
            for worker in [worker1 for worker1 in worker_list if worker1 not in selected_worker_list]:
                q_hat = q_bar + np.sqrt((K + 1) * np.log(np.sum(n_i)) / n_i)
                v = np.zeros(self.n)
                for worker1 in (selected_worker_list + [worker]):
                    v[worker1.id] = np.maximum(q_hat[worker1.id], v[worker1.id])
                ucb_diff = np.dot(w, v) - base
                # find the max heap
                criterion = - ucb_diff / worker.cost
                heapq.heappush(heap, (criterion, worker.id))
            # Recruit the worker with the maximum ratio of marginal UCB to cost.
            _, id = heapq.heappop(heap)
            selected_worker_list.append(worker_list[id])
        return selected_worker_list

    def online_policy_UWR(self, worker_list, budget=-1, K=-1):
        # Initialization:
        temp_worker_list = copy.deepcopy(worker_list)
        t = 1
        utility_dict = {}
        U = 0
        w = np.diff(sorted([random.uniform(0, 1) for _ in range(self.n + 1)]))  # weight
        u_ww = np.zeros(self.n)  # final completion quality of worker i
        q_bar = np.zeros(self.n)  # the average empirical quality value (reward) of worker i
        n_i = np.zeros(self.n)  # count for how many times each worker{i} has been learned
        # recruit all workers
        for worker in temp_worker_list:
            reward = worker.online_utility_list[t]
            reward_noise = reward + np.random.laplace(loc=0, scale=1 / worker.privacy_budget, size=1)[0]
            while (reward_noise < 0):
                reward_noise = reward + np.random.laplace(loc=0, scale=1 / worker.privacy_budget, size=1)[0]
            worker.reward_list.append(reward)
            worker.reward_noise_list.append(reward_noise)
            u_ww[worker.id] = worker.reward_noise_list[-1]
        # Add utility in this round to total record and deduct the cost
        U += np.dot(w, u_ww)
        selected_worker_list = temp_worker_list
        budget -= np.sum([worker.cost for worker in selected_worker_list])
        cardinality = np.array([1 if worker in selected_worker_list else 0 for worker in temp_worker_list])
        mask = np.isin(temp_worker_list, selected_worker_list)
        # update average quality value for each worker
        q_bar = np.where(mask, (q_bar * n_i + u_ww) / (n_i + cardinality), q_bar)
        n_i += cardinality
        utility_dict[t] = sum([i.reward_list[-1] for i in selected_worker_list])

        while True:
            t += 1
            selected_worker_list = self.select_winners(temp_worker_list, K, n_i, q_bar, w)
            if budget <= np.sum([worker.cost for worker in selected_worker_list]):
                sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
                return sum_utility, utility_dict
            # update worker profile
            for worker in temp_worker_list:
                reward = worker.online_utility_list[t]
                reward_noise = reward + np.random.laplace(loc=0, scale=1 / worker.privacy_budget, size=1)[0]
                while (reward_noise < 0):
                    reward_noise = reward + np.random.laplace(loc=0, scale=1 / worker.privacy_budget, size=1)[0]
                worker.reward_list.append(reward)
                worker.reward_noise_list.append(reward_noise)
                u_ww[worker.id] = worker.reward_noise_list[-1]
            U += np.dot(w, u_ww)
            budget -= np.sum([worker.cost for worker in selected_worker_list])
            cardinality = np.array([1 if worker in selected_worker_list else 0 for worker in temp_worker_list])
            mask = np.isin(temp_worker_list, selected_worker_list)
            q_bar = np.where(mask, (q_bar * n_i + u_ww) / (n_i + cardinality), q_bar)
            n_i += cardinality
            utility_dict[t] = sum([i.reward_list[-1] for i in selected_worker_list])

    # ----------------------------online evaluatation function-------------------------
    def run_off(self):
        dict_u = {}
        dict_d = {}
        for i in range(self.count):
            worker_list, task_list = self.loadDrons()
            self.worker_list = worker_list
            u1, d1 = self.online_policy_Greedy(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u2, d2 = self.online_policy_Random(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u3, d3 = self.online_policy_AUCB(worker_list=worker_list, budget=self.B, K=self.K,Time=self.T)
            u4, d4 = self.Baseline_CACI_OFF(worker_list=worker_list, budget=self.B, K=self.K)
            u5, d5 = self.online_policy_CACI_OFF(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u6, d6 = self.online_policy_UWR(worker_list=worker_list, budget=self.B, K=self.K)
            dict_u[i] = [u1, u2, u3, u4, u5, u6]
            dict_d[i] = [d1, d2, d3, d4, d5, d6]
        u1 = np.mean([dict_u[i][0] for i in dict_u.keys()])
        u2 = np.mean([dict_u[i][1] for i in dict_u.keys()])
        u3 = np.mean([dict_u[i][2] for i in dict_u.keys()])
        u4 = np.mean([dict_u[i][3] for i in dict_u.keys()])
        u5 = np.mean([dict_u[i][4] for i in dict_u.keys()])
        u6 = np.mean([dict_u[i][5] for i in dict_u.keys()])
        return [u1, u2, u3, u4, u5, u6]

    def run_on(self):
        dict_u = {}
        dict_d = {}
        for i in range(self.count):
            worker_list, task_list = self.loadDrons()
            self.worker_list = worker_list
            u1, d1 = self.online_policy_Greedy(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u2, d2 = self.online_policy_Random(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u3, d3 = self.Baseline_CACI_ON(worker_list=worker_list, budget=self.B, K=self.K)
            u4, d4 = self.online_policy_CACI_ON(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            dict_u[i] = [u1, u2, u3, u4]
            dict_d[i] = [d1, d2, d3, d4]
        u1 = np.mean([dict_u[i][0] for i in dict_u.keys()])
        u2 = np.mean([dict_u[i][1] for i in dict_u.keys()])
        u3 = np.mean([dict_u[i][2] for i in dict_u.keys()])
        u4 = np.mean([dict_u[i][3] for i in dict_u.keys()])
        return [u1, u2, u3, u4]

    def run_number_of_workers_off(self):
        dict_u = {}
        dict_d = {}
        for i in range(self.count):
            worker_list, task_list = self.loadDrons()
            self.worker_list = random.sample(worker_list,self.n)
            u1, d1 = self.online_policy_Greedy(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u2, d2 = self.online_policy_Random(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u3, d3 = self.online_policy_AUCB(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u4, d4 = self.Baseline_CACI_OFF(worker_list=worker_list, budget=self.B, K=self.K)
            u5, d5 = self.online_policy_CACI_OFF(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u6, d6 = self.online_policy_UWR(worker_list=worker_list, budget=self.B, K=self.K)
            dict_u[i] = [u1, u2, u3, u4, u5, u6]
            dict_d[i] = [d1, d2, d3, d4, d5, d6]
        u1 = np.mean([dict_u[i][0] for i in dict_u.keys()])
        u2 = np.mean([dict_u[i][1] for i in dict_u.keys()])
        u3 = np.mean([dict_u[i][2] for i in dict_u.keys()])
        u4 = np.mean([dict_u[i][3] for i in dict_u.keys()])
        u5 = np.mean([dict_u[i][4] for i in dict_u.keys()])
        u6 = np.mean([dict_u[i][5] for i in dict_u.keys()])
        return [u1, u2, u3, u4, u5, u6]

    def run_number_of_workers_on(self):
        dict_u = {}
        dict_d = {}
        for i in range(self.count):
            worker_list, task_list = self.loadDrons()
            self.worker_list = random.sample(worker_list,self.n)
            u1, d1 = self.online_policy_Greedy(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u2, d2 = self.online_policy_Random(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            u3, d3 = self.Baseline_CACI_ON(worker_list=worker_list, budget=self.B, K=self.K)
            u4, d4 = self.online_policy_CACI_ON(worker_list=worker_list, budget=self.B, K=self.K, Time=self.T)
            dict_u[i] = [u1, u2, u3, u4]
            dict_d[i] = [d1, d2, d3, d4]
        u1 = np.mean([dict_u[i][0] for i in dict_u.keys()])
        u2 = np.mean([dict_u[i][1] for i in dict_u.keys()])
        u3 = np.mean([dict_u[i][2] for i in dict_u.keys()])
        u4 = np.mean([dict_u[i][3] for i in dict_u.keys()])
        return [u1, u2, u3, u4]

    def run_K_OFF(self):
        self.reset()
        path_utility = os.path.join(result_address, 'CACI_OFF_K_utility' + '.txt')
        path_regret = os.path.join(result_address, 'CACI_OFF_K_regret' + '.txt')
        str = '%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % ('K', 'Greedy', 'Random', 'AUCB', 'Baseline_CACI_OFF', 'CACI_OFF', 'UWR')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        with open(path_regret, 'a') as f:
            f.write(str)

        K_list = [i for i in range(5, 16, 1)]
        for x in K_list:
            self.K = x
            utility_list = self.run_off()
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.K, utility_list[0], utility_list[1], utility_list[2], utility_list[3],
                utility_list[4], utility_list[5]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                self.K, utility_list[0], utility_list[1], utility_list[2], utility_list[3],
                utility_list[4], utility_list[5]))
            with open(path_regret, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.K, utility_list[3] - utility_list[0], utility_list[3] - utility_list[1], utility_list[3] - utility_list[2],
                    0, utility_list[3] - utility_list[4], utility_list[3] - utility_list[5]))

    def run_K_ON(self):
        self.reset()
        path_utility = os.path.join(result_address, 'CACI_ON_K_utility' + '.txt')
        path_regret = os.path.join(result_address, 'CACI_ON_K_regret' + '.txt')
        str = '%-18s%-18s%-18s%-18s%-18s\n' % ('K', 'Greedy', 'Random', 'Baseline_CACI_ON', 'CACI_ON')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        with open(path_regret, 'a') as f:
            f.write(str)

        K_list = [i for i in range(5, 16, 1)]
        for x in K_list:
            self.K = x
            utility_list = self.run_on()
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.K, utility_list[0], utility_list[1], utility_list[2], utility_list[3]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                self.K, utility_list[0], utility_list[1], utility_list[2], utility_list[3]))
            with open(path_regret, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.K, utility_list[2] - utility_list[0], utility_list[2] - utility_list[1], 0, utility_list[2] - utility_list[3]))

    def run_B_OFF(self):
        self.reset()
        path_utility = os.path.join(result_address, 'CACI_OFF_B_utility' + '.txt')
        path_regret = os.path.join(result_address, 'CACI_OFF_B_regret' + '.txt')
        str = '%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % ('B', 'Greedy', 'Random', 'AUCB', 'Baseline_CACI_OFF', 'CACI_OFF', 'UWR')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        with open(path_regret, 'a') as f:
            f.write(str)

        B_list = [i for i in range(1000, 10001, 1000)]
        for x in B_list:
            self.B = x
            utility_list = self.run_off()
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.B, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4], utility_list[5]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.B, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4], utility_list[5]))
            with open(path_regret, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.B, utility_list[3] - utility_list[0], utility_list[3] - utility_list[1], utility_list[3] - utility_list[2],
                    0, utility_list[3] - utility_list[4], utility_list[3] - utility_list[5]))

    def run_B_ON(self):
        self.reset()
        path_utility = os.path.join(result_address, 'CACI_ON_B_utility' + '.txt')
        path_regret = os.path.join(result_address, 'CACI_ON_B_regret' + '.txt')
        str = '%-18s%-18s%-18s%-18s%-18s\n' % ('B', 'Greedy', 'Random', 'Baseline_CACI_ON', 'CACI_ON')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        with open(path_regret, 'a') as f:
            f.write(str)

        B_list = [i for i in range(1000, 10001, 1000)]
        for x in B_list:
            self.B = x
            utility_list = self.run_on()
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.B, utility_list[0], utility_list[1], utility_list[2], utility_list[3]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.B, utility_list[0], utility_list[1], utility_list[2], utility_list[3]))
            with open(path_regret, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.B, utility_list[2] - utility_list[0], utility_list[2] - utility_list[1], 0,
                    utility_list[2] - utility_list[3]))

    def run_N_OFF(self):
        self.reset()
        path_utility = os.path.join(result_address, 'CACI_OFF_N_utility' + '.txt')
        path_regret = os.path.join(result_address, 'CACI_OFF_N_regret' + '.txt')
        str = '%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % ('N', 'Greedy', 'Random', 'AUCB', 'Baseline_CACI_OFF', 'CACI_OFF', 'UWR')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        with open(path_regret, 'a') as f:
            f.write(str)

        N_list = [i for i in range(25, 96, 5)]
        N_list=N_list + [99]
        for x in N_list:
            self.n = x
            utility_list = self.run_number_of_workers_off()
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.n, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4], utility_list[5]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.n, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4], utility_list[5]))
            with open(path_regret, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.n, utility_list[3] - utility_list[0], utility_list[3] - utility_list[1], utility_list[3] - utility_list[2],
                    0, utility_list[3] - utility_list[4], utility_list[3] - utility_list[5]))


    def run_N_ON(self):
        self.reset()
        path_utility = os.path.join(result_address, 'CACI_ON_N_utility' + '.txt')
        path_regret = os.path.join(result_address, 'CACI_ON_N_regret' + '.txt')
        str = '%-18s%-18s%-18s%-18s%-18s\n' % ('N', 'Greedy', 'Random', 'Baseline_CACI_ON', 'CACI_ON')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        with open(path_regret, 'a') as f:
            f.write(str)

        N_list = [i for i in range(25, 96, 5)]
        N_list=N_list + [99]
        for x in N_list:
            self.n = x
            utility_list = self.run_number_of_workers_on()
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.n, utility_list[0], utility_list[1], utility_list[2], utility_list[3]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.n, utility_list[0], utility_list[1], utility_list[2], utility_list[3]))
            with open(path_regret, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                    self.n, utility_list[2] - utility_list[0], utility_list[2] - utility_list[1], 0, utility_list[2] - utility_list[3]))

###########----------------------------Plot---------------------------------------##############
    def plot_regret_OFF_N(self):
        # plot N regret
        file_name = 'C:\\Users\\12282\\MINE\\Pycharm Fiile\\CACI\\CACI_result\\CACI_OFF_N_regret.txt'
        pdf_file = os.path.splitext(file_name)[0] + ".pdf"
        df = pd.read_fwf(file_name)
        columns_as_lists = {col: df[col].to_list() for col in df.columns}
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(columns_as_lists['N'], [i / 1000 for i in columns_as_lists['Random']], linestyle='-', color='dodgerblue',
                marker='s', markersize='16', linewidth=4, label='Random')
        ax.plot(columns_as_lists['N'], [i / 1000 for i in columns_as_lists['AUCB']], linestyle='-', color='khaki',
                marker='>', markersize='16', linewidth=4, label='AUCB')
        ax.plot(columns_as_lists['N'], [i / 1000 for i in columns_as_lists['CACI_OFF']], linestyle='-', color='salmon',
                marker='d', markersize='16', linewidth=4, label='CACI_OFF')
        ax.plot(columns_as_lists['N'], [i / 1000 for i in columns_as_lists['UWR']], linestyle='-', color='olivedrab',
                marker='v', markersize='16', linewidth=4, label='UWR')

        ax.set_ylabel(r"Regret($\times 10^3$)", fontsize=28)
        ax.set_xlabel("# of Total Workers", fontsize=28)
        ax.set_ylim(1, 9)
        ax.set_yticks(np.arange(2, 9, 2))
        ax.set_xlim(min(columns_as_lists['N']) - 1, max(columns_as_lists['N']) + 1)
        ax.set_xticks(columns_as_lists['N'][::3])
        ax.tick_params(axis='x', labelsize=28)
        ax.tick_params(axis='y', labelsize=28)
        ax.legend(loc='best', fontsize=24)
        ax.grid(linestyle='--', alpha=0.5)
        plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
        plt.show()

    def plot_regret_ON_N(self):
        # plot N regret
        file_name = 'C:\\Users\\12282\\MINE\\Pycharm Fiile\\CACI\\CACI_result\\CACI_ON_N_regret.txt'
        pdf_file = os.path.splitext(file_name)[0] + ".pdf"
        df = pd.read_fwf(file_name)
        columns_as_lists = {col: df[col].to_list() for col in df.columns}
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(columns_as_lists['N'], [i / 1000 for i in columns_as_lists['Random']], linestyle='-', color='dodgerblue',
                marker='s', markersize='16', linewidth=4, label='Random')
        ax.plot(columns_as_lists['N'], [i / 1000 for i in columns_as_lists['Greedy']], linestyle='-', color='gold',
                marker='>', markersize='16', linewidth=4, label='Greedy')
        ax.plot(columns_as_lists['N'], [i / 1000 for i in columns_as_lists['CACI_ON']], linestyle='-', color='red',
                marker='d', markersize='16', linewidth=4, label='CACI_ON')

        ax.set_ylabel(r"Regret($\times 10^3$)", fontsize=28)
        ax.set_xlabel("# of Total Workers", fontsize=28)
        ax.set_ylim(-0.5, 7)
        ax.set_yticks(np.arange(0, 6, 2))
        ax.set_xlim(min(columns_as_lists['N']) - 1, max(columns_as_lists['N']) + 1)
        ax.set_xticks(columns_as_lists['N'][::3])
        ax.tick_params(axis='x', labelsize=28)
        ax.tick_params(axis='y', labelsize=28)
        ax.legend(loc='best', fontsize=24)
        ax.grid(linestyle='--', alpha=0.5)
        plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
        plt.show()

    def plot_regret_OFF_B(self):
        # plot B regret
        file_name = 'C:\\Users\\12282\\MINE\\Pycharm Fiile\\CACI\\CACI_result\\CACI_OFF_B_regret.txt'
        pdf_file = os.path.splitext(file_name)[0] + ".pdf"
        df = pd.read_fwf(file_name)
        columns_as_lists = {col: df[col].to_list() for col in df.columns}
        fig, ax = plt.subplots(figsize=(10, 7))
        new_B_list = [i/1000 for i in columns_as_lists['B']]
        ax.plot(new_B_list, [i / 1000 for i in columns_as_lists['Random']], linestyle='-', color='dodgerblue',
                marker='s', markersize='16', linewidth=4, label='Random')
        ax.plot(new_B_list, [i / 1000 for i in columns_as_lists['AUCB']], linestyle='-', color='khaki',
                marker='>', markersize='16', linewidth=4, label='AUCB')
        ax.plot(new_B_list, [i / 1000 for i in columns_as_lists['CACI_OFF']], linestyle='-', color='salmon',
                marker='d', markersize='16', linewidth=4, label='CACI_OFF')
        ax.plot(new_B_list, [i / 1000 for i in columns_as_lists['UWR']], linestyle='-', color='olivedrab',
                marker='v', markersize='16', linewidth=4, label='UWR')

        ax.set_ylabel(r"Regret($\times 10^3$)", fontsize=28)
        ax.set_xlabel(r"Budget($\times 10^3$)", fontsize=28)
        ax.set_ylim(-1, 13)
        ax.set_yticks(np.arange(0, 13, 4))
        ax.set_xlim(0.8, 10.2)
        ax.set_xticks(new_B_list[::3])
        ax.tick_params(axis='x', labelsize=28)
        ax.tick_params(axis='y', labelsize=28)
        ax.legend(loc='best', fontsize=24)
        ax.grid(linestyle='--', alpha=0.5)
        plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
        plt.show()

    def plot_regret_ON_B(self):
        # plot B regret
        file_name = 'C:\\Users\\12282\\MINE\\Pycharm Fiile\\CACI\\CACI_result\\CACI_ON_B_regret.txt'
        pdf_file = os.path.splitext(file_name)[0] + ".pdf"
        df = pd.read_fwf(file_name)
        columns_as_lists = {col: df[col].to_list() for col in df.columns}
        fig, ax = plt.subplots(figsize=(10, 7))
        new_B_list = [i/1000 for i in columns_as_lists['B']]
        ax.plot(new_B_list, [i / 1000 for i in columns_as_lists['Random']], linestyle='-', color='dodgerblue',
                marker='s', markersize='16', linewidth=4, label='Random')
        ax.plot(new_B_list, [i / 1000 for i in columns_as_lists['Greedy']], linestyle='-', color='gold',
                marker='>', markersize='16', linewidth=4, label='Greedy')
        ax.plot(new_B_list, [i / 1000 for i in columns_as_lists['CACI_ON']], linestyle='-', color='red',
                marker='d', markersize='16', linewidth=4, label='CACI_ON')

        ax.set_ylabel(r"Regret($\times 10^3$)", fontsize=28)
        ax.set_xlabel(r"Budget($\times 10^3$)", fontsize=28)
        ax.set_ylim(-1, 10)
        ax.set_yticks(np.arange(0, 10, 3))
        ax.set_xlim(0.8,10.2)
        ax.set_xticks(new_B_list[::3])
        ax.tick_params(axis='x', labelsize=28)
        ax.tick_params(axis='y', labelsize=28)
        ax.legend(loc='best', fontsize=24)
        ax.grid(linestyle='--', alpha=0.5)
        plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
        plt.show()

    def plot_cumulative_utility_OFF_N(self):
        file_name = 'C:\\Users\\12282\\MINE\\Pycharm Fiile\\CACI\\CACI_result\\CACI_OFF_N_utility.txt'
        pdf_file = os.path.splitext(file_name)[0] + ".pdf"
        df = pd.read_fwf(file_name)
        df = df[df['N'].isin([25, 50, 75, 99])]
        cumulative_rewards = {col: df[col].to_list() for col in df.columns}
        del cumulative_rewards['N']
        cumulative_rewards = dict(sorted(cumulative_rewards.items(), key=lambda item: item[-1], reverse=True))
        cumulative_rewards = {key: [value / 10000 for value in values] for key, values in cumulative_rewards.items()}

        x = np.array([25, 50, 75, 100])
        width = 3.5  # 柱状宽度
        offsets = np.arange(-3.5, 7, 1.12) * width  # 柱状图的偏移
        colors = ['#f26c6a', '#b879e0', '#e8a746', '#5ba8ec', '#dec775', '#009b88']
        hatchs = ['//', '++', '\\', '--', '||', 'xx']
        plt.figure(figsize=(10, 7))
        for i, (label, data) in enumerate(cumulative_rewards.items()):
            plt.bar(x + offsets[i], data, width=width, color='white', edgecolor=colors[i], linewidth=3.5, label=label,
                    hatch=hatchs[i])

        plt.xlabel("# of Total Workers", fontsize=28)
        plt.ylabel(r"Cumulative utility($\times 10^3$)", fontsize=28)
        plt.xlim(8.5, 111)
        plt.xticks([25, 50, 75, 99])
        plt.ylim(0.25, 1.35)
        plt.yticks(np.arange(0.4, 1.3, 0.4))
        plt.tick_params(axis='x', labelsize=28)
        plt.tick_params(axis='y', labelsize=28)
        plt.legend(loc='upper left', ncol=3, frameon=False, bbox_to_anchor=(-0.01, 1.02),
                   columnspacing=0.2, handlelength=2, handletextpad=0.6, fontsize=21)
        plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
        plt.show()

    def plot_cumulative_utility_ON_N(self):
        file_name = 'C:\\Users\\12282\\MINE\\Pycharm Fiile\\CACI\\CACI_result\\CACI_ON_N_utility.txt'
        pdf_file = os.path.splitext(file_name)[0] + ".pdf"
        df = pd.read_fwf(file_name)
        df = df[df['N'].isin([25, 50, 75, 99])]
        cumulative_rewards = {col: df[col].to_list() for col in df.columns}
        del cumulative_rewards['N']
        cumulative_rewards = dict(sorted(cumulative_rewards.items(), key=lambda item: item[-1], reverse=True))
        cumulative_rewards = {key: [value / 10000 for value in values] for key, values in cumulative_rewards.items()}

        x = np.array([25, 50, 75, 100])
        width = 4.5  # 柱状宽度
        offsets = np.arange(-1.6, 7, 1.15) * width  # 柱状图的偏移
        colors = ['#f26c6a', '#b879e0', '#e8a746', '#5ba8ec', '#dec775', '#009b88']
        hatchs = ['//', '++', '\\', '--', '||', 'xx']
        plt.figure(figsize=(10, 7))
        for i, (label, data) in enumerate(cumulative_rewards.items()):
            plt.bar(x + offsets[i], data, width=width, color='white', edgecolor=colors[i], linewidth=3.5, label=label,
                    hatch=hatchs[i])

        plt.xlabel("# of Total Workers", fontsize=28)
        plt.ylabel(r"Cumulative utility($\times 10^3$)", fontsize=28)
        plt.xlim(12, 113)
        plt.xticks([25, 50, 75, 99])
        plt.ylim(0.25, 1.35)
        plt.yticks(np.arange(0.4, 1.3, 0.4))
        plt.tick_params(axis='x', labelsize=28)
        plt.tick_params(axis='y', labelsize=28)
        plt.legend(loc='upper left', ncol=2, frameon=False, bbox_to_anchor=(0.1, 1.02),
                   columnspacing=0.2, handlelength=2, handletextpad=0.6, fontsize=21)
        plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
        plt.show()

    def plot_cumulative_utility_OFF_B(self):
        file_name = 'C:\\Users\\12282\\MINE\\Pycharm Fiile\\CACI\\CACI_result\\CACI_OFF_B_utility.txt'
        pdf_file = os.path.splitext(file_name)[0] + ".pdf"
        df = pd.read_fwf(file_name)
        df = df[df['B'].isin([1000, 5000, 10000])]
        cumulative_rewards = {col: df[col].to_list() for col in df.columns}
        del cumulative_rewards['B']
        cumulative_rewards = dict(sorted(cumulative_rewards.items(), key=lambda item: item[-1], reverse=True))
        cumulative_rewards = {key: [value / 10000 for value in values] for key, values in cumulative_rewards.items()}

        x = np.array([0, 5, 10])
        width = 0.7  # 柱状宽度
        offsets = np.arange(-2.4, 8, 1.13) * width  # 柱状图的偏移
        colors = ['#ff4f4c', '#b879e0', '#e8a746', '#5ba8ec', '#dec775', '#009b88']
        hatchs = ['//', '++', '\\', '--', '||', 'xx']
        plt.figure(figsize=(10, 7))
        for i, (label, data) in enumerate(cumulative_rewards.items()):
            plt.bar(x + offsets[i], data, width=width, color='white', edgecolor=colors[i], linewidth=3.5, label=label,
                    hatch=hatchs[i])

        plt.xlabel(r"Budget($\times 10^3$)", fontsize=28)
        plt.ylabel(r"Cumulative utility($\times 10^3$)", fontsize=28)
        plt.xlim(-2.7, 13.5)
        plt.xticks([1, 5, 10])
        plt.ylim(0, 2.3)
        plt.yticks(np.arange(0.1, 2.3, 0.7))
        plt.tick_params(axis='x', labelsize=28)
        plt.tick_params(axis='y', labelsize=28)
        plt.legend(loc='upper left', ncol=3, frameon=False, bbox_to_anchor=(-0.01, 1.02),
                   columnspacing=0.2, handlelength=2, handletextpad=0.6, fontsize=21)
        plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
        plt.show()

    def plot_cumulative_utility_ON_B(self):
        file_name = 'C:\\Users\\12282\\MINE\\Pycharm Fiile\\CACI\\CACI_result\\CACI_ON_B_utility.txt'
        pdf_file = os.path.splitext(file_name)[0] + ".pdf"
        df = pd.read_fwf(file_name)
        df = df[df['B'].isin([1000, 5000, 10000])]
        cumulative_rewards = {col: df[col].to_list() for col in df.columns}
        del cumulative_rewards['B']
        cumulative_rewards = dict(sorted(cumulative_rewards.items(), key=lambda item: item[-1], reverse=True))
        cumulative_rewards = {key: [value / 10000 for value in values] for key, values in cumulative_rewards.items()}

        x = np.array([0, 5, 10])
        width = 0.9  # 柱状宽度
        offsets = np.arange(-1, 8, 1.12) * width  # 柱状图的偏移
        colors = ['#ff4f4c', '#b879e0', '#e8a746', '#5ba8ec', '#dec775', '#009b88']
        hatchs = ['//', '++', '\\', '--', '||', 'xx']
        plt.figure(figsize=(10, 7))
        for i, (label, data) in enumerate(cumulative_rewards.items()):
            plt.bar(x + offsets[i], data, width=width, color='white', edgecolor=colors[i], linewidth=3.5, label=label,
                    hatch=hatchs[i])

        plt.xlabel(r"Budget($\times 10^3$)", fontsize=28)
        plt.ylabel(r"Cumulative utility($\times 10^3$)", fontsize=28)
        plt.xlim(-2, 13.5)
        plt.xticks([1, 5, 10])
        plt.ylim(0, 2.3)
        plt.yticks(np.arange(0.1, 2.3, 0.7))
        plt.tick_params(axis='x', labelsize=28)
        plt.tick_params(axis='y', labelsize=28)
        plt.legend(loc='upper left', ncol=2, frameon=False, bbox_to_anchor=(0.1, 1.02),
                   columnspacing=0.2, handlelength=2, handletextpad=0.6, fontsize=21)
        plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
        plt.show()

if __name__ == '__main__':
    p = Policy()
    # #evaluate the impact of number of selected workers
    # p.run_K_OFF()
    # p.run_K_ON()
    # # -----evaluate the impact of number of total workers
    # p.run_N_OFF()
    # p.run_N_ON()
    # # -----evaluate the impact of budget
    # p.run_B_OFF()
    # p.run_B_ON()

    # -----plot the figures
    p.plot_regret_OFF_B()
    p.plot_regret_ON_B()
    p.plot_regret_OFF_N()
    p.plot_regret_ON_N()
    p.plot_cumulative_utility_OFF_N()
    p.plot_cumulative_utility_ON_N()
    p.plot_cumulative_utility_OFF_B()
    p.plot_cumulative_utility_ON_B()