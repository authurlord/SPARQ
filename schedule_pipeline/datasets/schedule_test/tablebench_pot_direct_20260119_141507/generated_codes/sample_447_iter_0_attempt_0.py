import pandas as pd

df = pd.read_csv('table.csv')
# Count frequency of each class
class_counts = df['Class'].value_counts()
# Identify classes that are not "10CR" or "10C" (the majority)
majority_classes = ['10CR', '10C']
abnormal_classes = class_counts[class_counts.index.isin(majority_classes) == False].index.tolist()
print(f"Final Answer: {abnormal_classes}")