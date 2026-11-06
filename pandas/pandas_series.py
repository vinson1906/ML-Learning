import numpy as np
import pandas as pd


# Create a Series for pandas

data = [3,4,6,7,8,0,10]

number_series = pd.Series(data)

print("Number Series:\n", number_series)

string_data = ["apple", "banana", "cherry", "grpes"]

string_series = pd.Series(string_data)
print("String Series:\n",string_series)

