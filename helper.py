import math

def gaussian_numerator(obs, mean, sd):
    """Implements probability density from Gaussian distribution

    Takes an observed value of one feature/variable existing in the training set
    and the sample mean and sample standard deviation of that variable as
    calculated in the training set. Plugs that data into the formula for
    normal/Gaussian distribution and returns a posterior probability density
    for
    observing this value for this feature under the assumption of a particular
    class label.

    An example would be: if we're using Naive Bayes to categorize a person as
    male or female, this would be a helper function that generates one part
    of the chain of independent feature posterior densities. Probability of
    male = gaussian_numerator(observedf1, mean of feature 1 in training set,
    sd of f1 in training set) multiplied across all features.
    """
    coefficient = 1 / ((2 * math.pi * sd**2)**(1/2))
    exponent = math.exp((-(obs - mean)**2)/(2 * sd**2))
    return coefficient * exponent
