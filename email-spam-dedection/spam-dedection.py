import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "Message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(df.head())

X = df['Message']
y = df['label']

# Split Data into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

# Convert Text Data into Numerical Form

vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Train the Naive Bayes Model

nb_model = MultinomialNB()
nb_model.fit(X_train_vector,y_train)


# Make Predictions on Test Data

y_pred = nb_model.predict(X_test_vector)

# Evaluate Model Performance

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy score is : {accuracy*100:.2f}%")

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham","Spam"],
            yticklabels=["Ham","Spam"])
plt.title("Confution Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test,y_pred))

