import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'tropical cyclones' to numeric, coercing errors to NaN if any (though none expected)
df['tropical cyclones'] = pd.to_numeric(df['tropical cyclones'], errors='coerce')
# Calculate the mean of the column
mean_cyclones = df['tropical cyclones'].mean()
print(f"Final Answer: {mean_cyclones:.1f}")