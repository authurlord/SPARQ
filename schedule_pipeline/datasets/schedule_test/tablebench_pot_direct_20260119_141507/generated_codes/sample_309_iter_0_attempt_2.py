import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract women's singles and mixed doubles champions
womens_singles = df['womens singles'].dropna().astype(str)
mixed_doubles = df['mixed doubles'].dropna().astype(str)

# Split mixed doubles entries (some are pairs, e.g., "janusz labisko anna zyśk")
mixed_doubles_list = []
for entry in mixed_doubles:
    if ',' in entry:
        names = [name.strip() for name in entry.split(',')]
        mixed_doubles_list.extend(names)
    else:
        mixed_doubles_list.append(entry.strip())

# Create a set of unique women's singles champions who also won mixed doubles
women_in_mixed = set()
for name in womens_singles:
    if name in mixed_doubles_list:
        women_in_mixed.add(name)

# Now count how many times each of these women won in women's singles
title_count = {}
for name in womens_singles:
    if name in women_in_mixed:
        title_count[name] = title_count.get(name, 0) + 1

# Find the champion with the most titles
if title_count:
    most_titles_champion = max(title_count, key=title_count.get)
    print(f"Final Answer: {most_titles_champion}")
else:
    print("Final Answer: none")