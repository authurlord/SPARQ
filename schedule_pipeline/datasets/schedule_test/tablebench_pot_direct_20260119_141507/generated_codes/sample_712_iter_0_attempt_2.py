import pandas as pd

df = pd.read_csv('table.csv')
# Sort by enrollment in descending order and get the institution with the highest enrollment
highest_enrollment_institution = df.loc[df['enrollment'].idxmax(), 'institution']
print(f"Final Answer: {highest_enrollment_institution}")