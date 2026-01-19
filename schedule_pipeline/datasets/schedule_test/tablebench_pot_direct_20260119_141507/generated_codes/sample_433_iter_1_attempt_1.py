import pandas as pd

df = pd.read_csv('table.csv')

# Basic insights
max_speakers = df['speakers'].max()
min_speakers = df['speakers'].min()
avg_speakers = df['speakers'].mean()
max_area = df.loc[df['speakers'].idxmax(), 'council area']
min_area = df.loc[df['speakers'].idxmin(), 'council area']

print(f"Final Answer: The council area with the most speakers is {max_area} ({max_speakers}), the least is {min_area} ({min_speakers}), and the average number of speakers is {avg_speakers:.0f}.")