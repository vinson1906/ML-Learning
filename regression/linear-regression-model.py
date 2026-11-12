import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#predict the car top speed with car age

age = [4,2,8,9,10,18,17,22,54,33,6,5]
top_speed = [140, 155, 110, 105,100, 87,89,76,46,66,133,130]

slope, intercept, r, p, std_err = stats.linregress(age,top_speed)

def predict_speed(x):
    return slope * x + intercept

prediction = predict_speed(15)

print("P:",p)
print("Relation: ", r)
print("Predicted car speed: ", round(prediction, 2))

plt.scatter(age,top_speed, color="blue", label="Actual values")

predictedSpeed = list(map(predict_speed,age))

plt.plot(age, predictedSpeed)

plt.scatter(15,prediction, color="green", label="predicted point" )
plt.savefig("car-speed-prediction.png")

print("saved file...!")
