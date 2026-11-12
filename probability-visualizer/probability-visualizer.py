import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, norm, uniform, expon


# Each option contains a SciPy distribution and values for plotting.
DISTRIBUTIONS = {
    "1": {
        "name": "Bernoulli Distribution",
        "dist": bernoulli(p=0.5),
        "discrete": True,
        "x": np.array([0, 1]),
    },
    "2": {
        "name": "Binomial Distribution",
        "dist": binom(n=10, p=0.5),
        "discrete": True,
        "x": np.arange(0, 11),
    },
    "3": {
        "name": "Poisson Distribution",
        "dist": poisson(mu=4),
        "discrete": True,
        "x": np.arange(0, 16),
    },
    "4": {
        "name": "Normal Distribution",
        "dist": norm(loc=50, scale=10),
        "discrete": False,
        "x": np.linspace(10, 90, 300),
    },
    "5": {
        "name": "Uniform Distribution",
        "dist": uniform(loc=0, scale=10),
        "discrete": False,
        "x": np.linspace(-1, 11, 300),
    },
    "6": {
        "name": "Exponential Distribution",
        "dist": expon(scale=2),
        "discrete": False,
        "x": np.linspace(0, 10, 300),
    },
}


def display_statistics(distribution, samples):
    """Print theoretical and sample statistics."""
    print("\n--- Distribution Statistics ---")
    print(f"Theoretical mean      : {distribution.mean():.3f}")
    print(f"Sample mean           : {np.mean(samples):.3f}")
    print(f"Theoretical variance  : {distribution.var():.3f}")
    print(f"Sample variance       : {np.var(samples):.3f}")
    print(f"Theoretical std. dev. : {distribution.std():.3f}")
    print(f"Sample std. dev.      : {np.std(samples):.3f}")


def visualize_distribution(choice):
    """Generate samples and plot PMF/PDF alongside the CDF."""
    config = DISTRIBUTIONS[choice]
    distribution = config["dist"]
    x_values = config["x"]
    is_discrete = config["discrete"]

    # Generate reproducible random samples.
    rng = np.random.default_rng(seed=42)
    samples = distribution.rvs(size=2000, random_state=rng)

    display_statistics(distribution, samples)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(config["name"], fontsize=16, fontweight="bold")

    if is_discrete:
        theoretical_pmf = distribution.pmf(x_values)
        sample_probabilities = np.array([
            np.mean(samples == value) for value in x_values
        ])

        width = 0.35
        axes[0].bar(
            x_values - width / 2,
            theoretical_pmf,
            width=width,
            label="Theoretical PMF",
        )
        axes[0].bar(
            x_values + width / 2,
            sample_probabilities,
            width=width,
            label="Sample probability",
        )
        axes[0].set_xlabel("Possible values")
        axes[0].set_ylabel("Probability")
        axes[0].set_title("Probability Mass Function (PMF)")
        axes[0].set_xticks(x_values)
    else:
        axes[0].hist(
            samples,
            bins=35,
            density=True,
            alpha=0.6,
            label="Generated samples",
        )
        axes[0].plot(
            x_values,
            distribution.pdf(x_values),
            linewidth=2,
            label="Theoretical PDF",
        )
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Probability density")
        axes[0].set_title("Probability Density Function (PDF)")

    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # Plot the cumulative distribution function.
    cdf_values = distribution.cdf(x_values)

    if is_discrete:
        axes[1].step(x_values, cdf_values, where="post")
        axes[1].scatter(x_values, cdf_values, s=20)
    else:
        axes[1].plot(x_values, cdf_values, linewidth=2)

    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_title("Cumulative Distribution Function (CDF)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def main():
    while True:
        print("\n========================================")
        print("   PROBABILITY DISTRIBUTION VISUALIZER")
        print("========================================")
        print("1. Bernoulli Distribution")
        print("2. Binomial Distribution")
        print("3. Poisson Distribution")
        print("4. Normal Distribution")
        print("5. Uniform Distribution")
        print("6. Exponential Distribution")
        print("0. Exit")

        choice = input("\nSelect a distribution (0-6): ").strip()

        if choice == "0":
            print("Thank you for using the visualizer!")
            break

        if choice not in DISTRIBUTIONS:
            print("Invalid choice. Please select a number from 0 to 6.")
            continue

        visualize_distribution(choice)


if __name__ == "__main__":
    main()
