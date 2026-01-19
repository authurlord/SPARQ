import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'tropical cyclones' column to numeric for calculation
df['tropical cyclones'] = pd.to_numeric(df['tropical cyclones'])
# Calculate the average number of tropical cyclones per season
average_cyclones = df['tropical cyclones'].mean()
print(f"Final Answer: {average_cyclones:.1f}")