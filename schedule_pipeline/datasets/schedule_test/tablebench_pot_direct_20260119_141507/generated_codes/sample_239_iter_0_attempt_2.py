import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between typhus and typhoid fever
correlation = df['typhus'].corr(df['typhoid fever'])
print(f"Final Answer: {correlation:.3f}")