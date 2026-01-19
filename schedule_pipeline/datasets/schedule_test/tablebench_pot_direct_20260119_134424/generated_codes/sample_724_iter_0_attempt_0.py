import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'Power output (hp)' column by extracting the maximum value for ranges like '200/235'
def extract_power(power_str):
    if '/' in power_str:
        return max(map(int, power_str.split('/')))
    return int(power_str)

df['Power output (hp)'] = df['Power output (hp)'].apply(extract_power)

# Sort by power output in descending order
ranked_models = df.sort_values(by='Power output (hp)', ascending=False)['Model']

print(f"Final Answer: {', '.join(ranked_models)}")