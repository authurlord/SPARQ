import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Number' column to numeric, errors='coerce' will turn invalid parsing into NaN
df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
# Count rows where Number > 1500
count = df[df['Number'] > 1500].shape[0]
print(f"Final Answer: {count}")