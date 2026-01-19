import pandas as pd

df = pd.read_csv('table.csv')
# Remove rows where 'P' is missing or empty
df_filtered = df[df['P'].notna() & (df['P'] != '-')]
# Calculate the mean of 'P' column
mean_p = df_filtered['P'].astype(float).mean()
print(f"Final Answer: {mean_p:.1f}")