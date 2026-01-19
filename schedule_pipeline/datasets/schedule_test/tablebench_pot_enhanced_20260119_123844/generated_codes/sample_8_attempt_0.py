import pandas as pd

df = pd.read_csv('table.csv')
# Remove the '%' sign and convert to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)
# Calculate the average
average_national_vote = df['% of national vote'].mean()
print(f"Final Answer: {average_national_vote:.2f}")