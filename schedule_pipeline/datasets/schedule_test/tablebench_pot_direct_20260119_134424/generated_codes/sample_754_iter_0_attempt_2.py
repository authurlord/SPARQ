import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('table.csv')
df['qual'] = pd.to_numeric(df['qual'])
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['qual'], marker='o', linestyle='-', color='b')
correlation = df['qual'].corr(df['finish'])
print(f"Correlation between qualifying speed and finishing position: {correlation:.2f}")