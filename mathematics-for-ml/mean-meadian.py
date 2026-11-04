import numpy as np
from scipy import stats

array = [828,2828,828,2765,7675,7373,8594,975]
result = np.mean(array)
print(result);
median = np.median(array)
print(median)

my_mode = stats.mode(array)
print(my_mode)





