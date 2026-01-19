import pandas as pd

df = pd.read_csv('table.csv')
# Count the frequency of each class
class_counts = df['Class'].value_counts()
# Identify classes that appear only once (potential outliers)
abnormal_classes = class_counts[class_counts == 1].index.tolist()
print(f"Final Answer: {', '.join(abnormal_classes)}")