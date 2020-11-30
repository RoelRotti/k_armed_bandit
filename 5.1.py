import numpy, math, scipy.stats, norm
import matplotlib.pyplot as plt

k = 15  # 5 <| k <| 20
T = 1000  # Number of actions


def k_bandit(number_arms, number_actions, e_value, arms):

    average_reward = []
    total_reward = []
    times_pulled = []
    total_regret = []
    upper_bound = []

    for arm in range(number_arms):
        average_reward.append(0)
        total_reward.append(0)
        times_pulled.append(0)
        upper_bound.append(0)

    for t in range(number_actions):
        total_regret.append(0)

    # For 'Number of actions', do

    for time in range(number_actions):

        # Best strategy
        max_value = max(average_reward)
        exploitation = int(average_reward.index(max_value))

        # e-greedy
        if e_value != -1:
            # Pick strategy: e-greedy, UCB
            if numpy.random.uniform(0, 1) > e_value:  # greedy exploitation
                total_reward, times_pulled, average_reward, upper_bound, total_regret = draw_arm(exploitation, arms, total_reward, times_pulled, average_reward, upper_bound, total_regret, time)
            else:  # Exploration
                exploration = int(numpy.random.uniform(0, k - 1))  # TODO: implement better drawing
                while exploration == exploitation:
                    exploration = int(numpy.random.uniform(0, k - 1))
                total_reward, times_pulled, average_reward, upper_bound, total_regret = draw_arm(exploration, arms, total_reward, times_pulled, average_reward, upper_bound, total_regret, time)
        # UCB
        else:
            # Explore each arm 10 times: hardcoded
            if times_pulled[number_arms-1] < 15:
                for arm in range(number_arms):
                    if times_pulled[arm] < 15:
                        total_reward, times_pulled, average_reward, upper_bound, total_regret = draw_arm(arm, arms, total_reward,
                                                                                       times_pulled, average_reward,
                                                                                       upper_bound, total_regret, time)
                        break
            else:
                max_upper_bound = max(upper_bound)
                arm_max_upper_bound = int(upper_bound.index(max_upper_bound))
                total_reward, times_pulled, average_reward, upper_bound, total_regret = draw_arm(arm_max_upper_bound, arms, total_reward, times_pulled, average_reward, upper_bound, total_regret, time)

    # Compensate for total regret
    optimum = max(average_reward)
    for times in range(T):
        total_regret[times] += (times+1) * optimum

    return total_regret


# Create + save k-arms (random mu)
def draw_arm(arm, arms, total_reward, times_pulled, average_reward, upper_bound, total_regret, time):
    mu = arms[arm]
    value = numpy.random.normal(mu, 1)
    total_reward[arm] += value
    times_pulled[arm] += 1
    average_reward[arm] = total_reward[arm] / times_pulled[arm]
    #Calculate upper bound for UCB
    if value > upper_bound[arm]:
        upper_bound[arm] = value
    # Calculate L(t) + save to list
    # How to calculate total_regret:
    # total_regret += (optimum - average_reward[arm])
    # But optimum not known so:
    if value > 0:
        t = time
        while t < T:
            total_regret[t] -= value  # at the end: add k*optimum value
            t = t + 1
    else:
        t = time
        while t < T:
            total_regret[t] += value  # at the end: add k*optimum value
            t = t + 1

    return total_reward, times_pulled, average_reward, upper_bound, total_regret


arms = []
for arm in range(k):
    arms.append(numpy.random.uniform(0, 20))  # self defined borders


k1 = k_bandit(k, T, 0.1, arms)
k2 = k_bandit(k, T, 0.01, arms)
k3 = k_bandit(k, T, -1, arms)

x = numpy.linspace(1,1000,1000)
log = 50*numpy.log10(x)
plt.plot(x, log)

# Plot L(t)
plt.plot(k1, label = 'e-greedy: 0.1')
plt.plot(k2, label = 'e-greedy: 0.01')
plt.plot(k3, label = 'UCB')

plt.xlabel("Time(t)")
plt.ylabel("Total regret L(t)")
plt.title("Total regret for k-armed bandit problem (k = 14, bounds mean: (-10,10))")
plt.legend()
#plt.ylim(0)
plt.show()
# plt.ylabel('some numbers')
plt.show()
