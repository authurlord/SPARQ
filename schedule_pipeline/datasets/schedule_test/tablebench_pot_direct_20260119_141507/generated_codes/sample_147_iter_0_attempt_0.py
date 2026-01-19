import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
examinees = df['number of examinees'].astype(float)
pass_percentage = df['pass percentage'].str.replace('%', '').astype(float)

# Calculate the correlation
correlation = examinees.corr(pass_percentage)
print(f"Final Answer: {correlation:.3f}")