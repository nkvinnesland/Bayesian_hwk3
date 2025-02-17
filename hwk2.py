import numpy as np
import matplotlib.pyplot as plt


prior_array = [False] * 990 + [True] * 10
likelihood_array = [True] * 95 + [False] * 5
not_likelihood_array = [True] * 5 + [False] * 95
pop_size = 10000

def perform_calcs(population, pr_array, li_array, not_li_array):
    samples = np.random.choice(pr_array, population)
    results = []
    for diseased in samples:
        if diseased:
            results.append(np.random.choice(li_array))
        else:
            results.append(np.random.choice(not_li_array))
    
    TP = sum(1 for i in range(population) if samples[i] and results[i])
    FP = sum(1 for i in range(population) if not samples[i] and results[i])

    total_p = TP + FP
    posterior_prob = TP/total_p

    print(f"True positive rate: {TP}")
    print(f"False positive rate: {FP}")
    print(f"Posterior Probability: {posterior_prob:.2f}%")

    return posterior_prob

perform_calcs(pop_size, prior_array, likelihood_array, not_likelihood_array)
print()

# After reading up on numpy a bit I am going to use the np.random.normal(mean, std) method and np.clip to create different prior/TPR/FPR values.
# np.random.normal(mean, std) uses a normal (gaussian) distribution.
# because of this, we may see values that are negative or above 1 (due to 68-95-99.7 empircal rule) so we will clip them.

k = 1000

posterior_probs_arr = []

for _ in range(k):
    prior_probability = np.clip(np.random.normal(0.01, 0.005), 0 , 1)
    t_positive_rate = np.clip(np.random.normal(0.95, 0.01), 0, 1)
    f_positive_rate = np.clip(np.random.normal(0.05, 0.01), 0, 1)

    prior = int(pop_size * prior_probability) # need to cast to int because number of people needs to be a whole number
    new_p_array = [False] * (pop_size-prior) + [True] * prior
    new_t_p_arr = [True] * int(100 * t_positive_rate) + [False] * int(100 - (t_positive_rate * 100))
    new_f_p_arr = [True] * int(100 * f_positive_rate) + [False] * int(100 - (f_positive_rate * 100))

    print(f"Iteration: {_+1}")
    posterior_probs_arr.append(perform_calcs(pop_size, new_p_array, new_t_p_arr, new_f_p_arr))
    print()

posterior_probs_arr = np.array(posterior_probs_arr)

from collections import Counter

mean = np.mean(posterior_probs_arr)
median = np.median(posterior_probs_arr)
std = np.std(posterior_probs_arr)

#round posterior probs to 2 decimals
rounded_probs = np.round(posterior_probs_arr, 2)
counts = Counter(rounded_probs)
mode = max(counts, key = counts.get)

print(f"Mean value: {mean}")
print(f"Median value: {median}")
print(f"Standard deviation: {std}")
print(f"Mode value: {mode:.2f}")

# plt.figure(figsize=(10, 10))
# plt.hist(posterior_probs_arr, bins=3, edgecolor="black", alpha=0.7)
# plt.axvline(x=0.16, color="red", linestyle="dashed", linewidth=4, label="Analytical Value")
# plt.axvline(x=mean, color="blue", linestyle="dotted", linewidth=3, label="Mean Value")
# plt.axvline(x=median, color="yellow", linestyle="dashdot", linewidth=2, label="Median Value")
# plt.axvline(x=mode, color="orange", linestyle="solid", linewidth=1, label="Mode Value")

# plt.xlabel("Posterior Probability")
# plt.ylabel("Frequency")
# plt.title("Homework 3")
# plt.legend()
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.show()

from scipy.stats import norm


# Plot Histogram with Gaussian Fit
plt.figure(figsize=(10, 6))

# Histogram with density=True to match Gaussian scaling
count, bins, _ = plt.hist(posterior_probs_arr, bins=3, edgecolor="black", alpha=0.7, density=True)

# Generate x values for Gaussian curve
x_values = np.linspace(min(posterior_probs_arr), max(posterior_probs_arr), 100)

# Compute Gaussian curve using mean and std
gaussian_curve = norm.pdf(x_values, mean, std)

# Plot the Gaussian curve
plt.plot(x_values, gaussian_curve, color="red", linewidth=2, label="Gaussian Fit")

# Add reference lines
plt.axvline(x=0.16, color="red", linestyle="dashed", linewidth=4, label="Analytical Value")
plt.axvline(x=mean, color="blue", linestyle="dotted", linewidth=3, label="Mean Value")
plt.axvline(x=median, color="yellow", linestyle="dashdot", linewidth=2, label="Median Value")
plt.axvline(x=mode, color="orange", linestyle="solid", linewidth=1, label="Mode Value")

# Labels and Title
plt.xlabel("Posterior Probability")
plt.ylabel("Density")
plt.title("Posterior Probabilities with Gaussian Fit")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the final plot
plt.show()