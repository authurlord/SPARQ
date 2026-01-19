import pandas as pd

df = pd.read_csv('table.csv')
# Count the frequency of each value in the 'Class' column
class_counts = df['Class'].value_counts()
# Identify values that appear only once (potential outliers)
abnormal_values = class_counts[class_counts == 1].index.tolist()
print(f"Final Answer: {abnormal_values}")