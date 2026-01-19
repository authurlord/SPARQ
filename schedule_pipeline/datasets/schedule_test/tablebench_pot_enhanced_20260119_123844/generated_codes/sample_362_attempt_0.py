import pandas as pd

df = pd.read_csv('table.csv')
# Extract melting point (first value before '/')
melting_points = df['melting / boiling point'].str.split('/').str[0]
# Remove non-numeric characters and convert to float
melting_points = pd.to_numeric(melting_points.str.replace(r'[^\d\.-]', '', regex=True), errors='coerce')
# Count agents with melting point below 0
count_below_zero = (melting_points < 0).sum()
print(f"Final Answer: {count_below_zero}")