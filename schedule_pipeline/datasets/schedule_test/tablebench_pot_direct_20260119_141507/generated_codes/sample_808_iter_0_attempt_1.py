import pandas as pd

df = pd.read_csv('table.csv')
# Remove rows where 'P' is missing or null (e.g., '-')
valid_p_values = df[df['P'] != '-']['P'].astype(float)
mean_p = valid_p_values.mean()
print(f"Final Answer: {mean_p:.1f}")