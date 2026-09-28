import numpy as np


def cross_entropy(probabilities, label):
    return -np.log(probabilities[label])


probabilities = np.array([
    0.05,
    0.10,
    0.70,
    0.03,
    0.02,
    0.02,
    0.03,
    0.02,
    0.01,
    0.02
])

label = 2

loss = cross_entropy(probabilities, label)

print("label =", label)
print("probability =", probabilities[label])
print("loss =", loss)