import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['pleasure'] = pd.to_numeric(df['pleasure'])
df['psychological dependence'] = pd.to_numeric(df['psychological dependence'])
df['physical dependence'] = pd.to_numeric(df['physical dependence'])

# Calculate correlation between pleasure and psychological dependence
corr_psy = df['pleasure'].corr(df['psychological dependence'])

# Calculate correlation between pleasure and physical dependence
corr_phy = df['pleasure'].corr(df['physical dependence'])

# Compare correlations
if corr_psy > corr_phy:
    result = "psychological dependence"
else:
    result = "physical dependence"

print(f"Final Answer: {result}")