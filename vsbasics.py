import numpy as np
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt

x = random.uniform(0.0, 1.0, 500)
epsilon = random.uniform(0,1)
print(epsilon)
print(len([i for i in x if i < epsilon]))
sns.kdeplot(x)
plt.show()