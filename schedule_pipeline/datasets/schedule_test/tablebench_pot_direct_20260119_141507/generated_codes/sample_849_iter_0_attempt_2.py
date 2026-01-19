import pandas as pd

df = pd.read_csv('table.csv')
# Extract enrollment values
enrollments = df['enrollment'].astype(int)
max_enrollment = enrollments.max()
min_enrollment = enrollments.min()
difference = max_enrollment - min_enrollment
print(f"Final Answer: {difference}")