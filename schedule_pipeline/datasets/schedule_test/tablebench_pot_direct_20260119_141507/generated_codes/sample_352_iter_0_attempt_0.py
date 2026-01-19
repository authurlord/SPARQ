import pandas as pd

df = pd.read_csv('table.csv')
# Count universities located in London
london_universities = df[df['location'] == 'london'].shape[0]
print(f"Final Answer: {london_universities}")