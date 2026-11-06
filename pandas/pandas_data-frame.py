import pandas as pd

# create a data frame 

sample = ["one","two","three","four"]

df = pd.DataFrame(sample)

print("Sample Data Frame:\n",df)

#Read the data from Excel file
sd = pd.read_csv("student_data.csv")

#Print first five datas
print(sd.head())

#show the datas details like data type and column
sd.info()

#fill 0 if a data is null

print(sd.isnull().sum())
sd = sd.fillna(0)

#filter with age

vote_aligible = sd[sd['age'] >= 18]
print("Vote aligible students are :\n", vote_aligible)

#add new column

sd["total"] = sd["tamil"] + sd["English"]
print(sd[sd['total'] >= 130])

group = sd.groupby("category")["sales"].sum()

print("this mounth sales:\n ",group)