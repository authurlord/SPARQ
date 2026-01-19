import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'enrollment' column to integer type
df['enrollment'] = pd.to_numeric(df['enrollment'])
# Find the institution with the highest enrollment
max_enrollment_institution = df.loc[df['enrollment'].idxmax(), 'institution']
print(f"Final Answer: {max_enrollment_institution}")