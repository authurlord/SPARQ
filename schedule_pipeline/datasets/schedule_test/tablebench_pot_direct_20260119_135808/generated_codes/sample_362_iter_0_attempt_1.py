import pandas as pd

df = pd.read_csv('table.csv')

# Extract melting point from the 'melting / boiling point' column
# Split by '/' and take the first part (melting point)
df['melting_point'] = df['melting / boiling point'].str.split('/').str[0]

# Remove non-numeric characters and convert to float
df['melting_point'] = df['melting_point'].str.replace('[^0-9\-]', '', regex=True).astype(float)

# Count agents with melting point below 0
count_below_zero = (df['melting_point'] < 0).sum()

print(f"Final Answer: {count_below_zero}")