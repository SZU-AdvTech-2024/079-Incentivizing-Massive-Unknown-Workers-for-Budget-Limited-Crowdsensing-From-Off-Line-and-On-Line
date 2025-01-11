dataset_address = 'C:\\Users\\12282\\Desktop\\coding\\Data'
result_address = 'C:\\Users\\12282\\Desktop\\coding\\Result\\CACI'

T = 6000
n = 50 # of all workers
m = 20 # of all tasks
size = 1
B = 5000
K = 5 # of picked workers
count = 20
privacy_budget = 1 # privacy budget
variance = 0.1
alpha_step=10
b_max = 1
epsilon = 0.1

def logistic_map(x, r=3.9, n=100):
    result = []
    for _ in range(n):
        x = r * x * (1 - x)
        result.append(x)
    return result


