import pandas as pd

df = pd.read_csv('table.csv')
# Count frequency of each class
class_counts = df['Class'].value_counts()
# Identify classes with low frequency (less than 2 occurrences)
abnormal_classes = class_counts[class_counts < 2].index.tolist()
print(f"Final Answer: {', '.join(abnormal_classes)}")