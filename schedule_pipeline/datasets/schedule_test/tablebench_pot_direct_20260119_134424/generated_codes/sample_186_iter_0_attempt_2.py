import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('table.csv')
df['of candidates nominated'] = pd.to_numeric(df['of candidates nominated'])
df['% of popular vote'] = pd.to_numeric(df['% of popular vote'])
plt.figure(figsize=(10, 6))
plt.plot(df['election'], df['% of popular vote'], marker='o', label='% of Popular Vote')
plt.plot(df['election'], df['of candidates nominated'], marker='s', label='Number of Candidates Nominated')
plt.xticks(rotation=45)