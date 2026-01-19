import pandas as pd

df = pd.read_csv('table.csv')
# Filter municipalities where Spanish speakers >= 40,000
spanish_40k_plus = df[df['spanish'] >= 40000]
# Count the number of such municipalities
count = len(spanish_40k_plus)
print(f"Final Answer: {count}")