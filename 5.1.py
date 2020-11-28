import numpy, math, scipy.stats, norm
import matplotlib.pyplot as plt

k = 15  # 5 <| k <| 20
T = 1000  # Number of actions
e = 0.1

arms = []
average_reward = []
total_reward = []
times_pulled = []
total_regret = []


# Create + save k-arms (random mu)
def draw_arm(arm):  # , arms, total_reward, times_pulled, average_reward):
    mu = arms[arm]
    value = numpy.random.normal(mu, 1)
    total_reward[arm] += value
    times_pulled[arm] += 1
    average_reward[arm] = average_reward[arm] / times_pulled[arm]


def k_bandit(number_arms, number_actions, e_value):
    for arm in range(number_arms):
        arms.append(numpy.random.uniform(-50, 50))  # self defined borders
        average_reward.append(0)
        total_reward.append(0)
        times_pulled.append(0)
    # For 'Number of actions', do
    for time in range(number_actions):
        # Best strategy
        max_value = max(average_reward)
        exploitation = int(average_reward.index(max_value))

        # Pick strategy: e-greedy, UCB
        if numpy.random.uniform(0, 1) > e_value:  # greedy exploitation
            draw_arm(exploitation)  # , arms, total_reward, times_pulled, average_reward):
        else:  # Exploration
            exploration = int(numpy.random.uniform(0, k - 1))  # TODO: implement better drawing
            while exploration == exploitation:
                exploration = int(numpy.random.uniform(0, k - 1))
            draw_arm(exploration)  # , arms, total_reward, times_pulled, average_reward):

        # Calculate L(t) + save to list
        total_regret.append(0)  # add new time to the list
        for arm in range(number_arms):
            # How to calculate total_regret:
            # total_regret += (optimum - average_reward[arm])
            # But optimum not known so:
            total_regret[time] -= average_reward[arm]  # at the end: add k*optimum value


k_bandit(k, T, e)

# Compensate for total regret
optimum = max(average_reward)
for times in range(T):
    total_regret[times] += k*optimum


# Plot L(t)
plt.plot(total_regret)
#plt.ylabel('some numbers')
plt.show()
