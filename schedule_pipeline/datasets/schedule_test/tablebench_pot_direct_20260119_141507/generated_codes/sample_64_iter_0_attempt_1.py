import pandas as pd

df = pd.read_csv('table.csv')
# Filter nations with 1 gold, 0 silver, 0 bronze
result = df[(df['gold'] == 1) & (df['silver'] == 0) & (df['bronze'] == 0)]
# Get the nation name
nation = result['nation'].iloc[0] if not result.empty else None
print(f"Final Answer: {nation}")