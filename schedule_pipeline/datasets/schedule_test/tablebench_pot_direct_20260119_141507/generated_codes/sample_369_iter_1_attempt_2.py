import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'semifinalists' column: remove 't' suffix and convert to int
def clean_semifinalists(x):
    if isinstance(x, str) and 't' in x:
        return int(x.replace('t', ''))
    elif isinstance(x, (int, float)):
        return int(x)
    else:
        return 0

df['semifinalists'] = df['semifinalists'].apply(clean_semifinalists)

# Count how many countries have at least one semifinalist (>= 1)
count = (df['semifinalists'] >= 1).sum()

print(f"Final Answer: {count}")