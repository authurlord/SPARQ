import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'Power output (hp)' column: replace '200/235' with 235, and convert to float
def clean_power_output(x):
    if isinstance(x, str) and '/' in x:
        return float(x.split('/')[-1].strip())
    elif isinstance(x, str):
        return float(x.strip())
    else:
        return x

df['Power output (hp)'] = df['Power output (hp)'].apply(clean_power_output)

# Sort by power output in descending order and get the rank
ranked_models = df[['Model', 'Power output (hp)']].sort_values(by='Power output (hp)', ascending=False)

# Extract model names in ranked order
final_ranking = ranked_models['Model'].tolist()

print(f"Final Answer: {', '.join(final_ranking)}")