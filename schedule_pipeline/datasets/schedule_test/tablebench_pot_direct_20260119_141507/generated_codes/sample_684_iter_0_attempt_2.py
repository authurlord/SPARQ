import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'typhus' and 'smallpox' columns
typhus_cases = df['typhus']
smallpox_cases = df['smallpox']

# Calculate the correlation coefficient
correlation_coefficient = typhus_cases.corr(smallpox_cases)
print(f"Final Answer: {correlation_coefficient:.2f}")