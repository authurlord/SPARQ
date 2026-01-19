import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'occurrence' column to integer to enable numerical comparison
df['occurrence'] = pd.to_numeric(df['occurrence'], errors='coerce')
# Count rows where occurrence > 1
count = df[df['occurrence'] > 1].shape[0]
print(f"Final Answer: {count}")