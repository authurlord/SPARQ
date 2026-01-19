import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'tropical cyclones' to integer type to avoid conversion errors
df['tropical cyclones'] = pd.to_numeric(df['tropical cyclones'], errors='coerce')
# Calculate the mean of tropical cyclones per season
mean_cyclones = df['tropical cyclones'].mean()
print(f"Final Answer: {mean_cyclones:.1f}")