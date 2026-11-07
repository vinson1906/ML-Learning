import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x = np.random.uniform(1,5,1000)
y= np.random.uniform(2,10,1000)
#histogram
plt.hist(x, bins=20)
plt.savefig("histogram.png")
plt.close()

print("File saved successfully...!")

#scatter plot

plt.scatter(x,y,color="orange", label="Scatter Plot Example")
plt.savefig("scatter.png")

print("file saved..!")
