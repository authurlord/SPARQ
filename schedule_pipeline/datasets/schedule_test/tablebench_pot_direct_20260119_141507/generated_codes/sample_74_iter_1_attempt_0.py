import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where Total is 8 and Gold is 3
result = df[(df['Total'] == 8) & (df['Gold'] == 3)]

# Extract the Nation name
nation = result['Nation'].values[0] if not result.empty else None

print(f"Final Answer: {nation}")