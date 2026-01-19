import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'mean', 'pleasure', etc. to numeric (they are already numbers, but ensure correct type)
df['pleasure'] = pd.to_numeric(df['pleasure'])
df['psychological dependence'] = pd.to_numeric(df['psychological dependence'])
df['physical dependence'] = pd.to_numeric(df['physical dependence'])

# Calculate correlation between pleasure and each dependence
corr_psych = df['pleasure'].corr(df['psychological dependence'])
corr_physical = df['pleasure'].corr(df['physical dependence'])

print(f"Correlation with psychological dependence: {corr_psych:.3f}")
print(f"Correlation with physical dependence: {corr_physical:.3f}")

# Determine which has a greater impact (higher absolute correlation)
if abs(corr_psych) > abs(corr_physical):
    final_answer = "psychological"
else:
    final_answer = "physical"

print(f"Final Answer: {final_answer}")