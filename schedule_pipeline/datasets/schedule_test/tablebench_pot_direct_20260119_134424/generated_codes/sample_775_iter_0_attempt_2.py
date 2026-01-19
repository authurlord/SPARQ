import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to numeric by removing '%' and converting to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)
# Compute average success rate over the period
avg_success_rate = df['success rate'].mean()
print(f"Final Answer: {avg_success_rate:.1f}")