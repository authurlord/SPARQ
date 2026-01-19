import pandas as pd

df = pd.read_csv('table.csv')
max_enrollment = df['enrollment'].max()
min_enrollment = df['enrollment'].min()
difference = max_enrollment - min_enrollment
print(f"Final Answer: {difference}")