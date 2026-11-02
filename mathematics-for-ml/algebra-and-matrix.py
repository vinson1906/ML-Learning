import numpy as np

A = np.array([
    [7,8,8],
    [9,7,5],
    [9,5,6]
])

B = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

result = np.dot(A,B)

print("result:",result)

reshaped = result.reshape(9,1)
print("reshaped:\n",reshaped)

A.resize(2,9)
print("A after resize:\n",A)

print("A first element", A[0][0])