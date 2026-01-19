import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'tropical cyclones' column to numeric, coercing errors to NaN (though none expected)
df['tropical cyclones'] = pd.to_numeric(df['tropical cyclones'], errors='coerce')
# Calculate the mean of the tropical cyclones column
mean_cyclones = df['tropical cyclones'].mean()
print(f"Final Answer: {mean_cyclones:.1f}")