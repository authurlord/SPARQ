import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
typhus = df['typhus']
typhoid_fever = df['typhoid fever']

# Calculate the correlation coefficient
correlation = typhus.corr(typhoid_fever)
print(f"Final Answer: {correlation:.2f}")