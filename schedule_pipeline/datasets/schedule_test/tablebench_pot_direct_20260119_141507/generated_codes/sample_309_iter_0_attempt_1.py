import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract women's singles champions and mixed doubles champions
women_singles = df['womens singles'].dropna().tolist()
mixed_doubles = df['mixed doubles'].dropna().tolist()

# Find women's singles champions who also won mixed doubles
common_champions = []
for name in women_singles:
    if any(name in md for md in mixed_doubles):
        common_champions.append(name)

# Count occurrences of each common champion in women's singles
from collections import Counter
champion_counts = Counter(common_champions)

# Find the champion with the maximum count
if champion_counts:
    most_titles = champion_counts.most_common(1)[0][0]
    print(f"Final Answer: {most_titles}")
else:
    print("Final Answer: none")