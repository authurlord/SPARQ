import pandas as pd

df = pd.read_csv('table.csv')
# Filter nations with 1 gold, 0 silver, and 0 bronze
result = df[(df['gold'] == 1) & (df['silver'] == 0) & (df['bronze'] == 0)]
# Extract the nation name
nation = result['nation'].values[0] if not result.empty else None
print(f"Final Answer: {nation}")