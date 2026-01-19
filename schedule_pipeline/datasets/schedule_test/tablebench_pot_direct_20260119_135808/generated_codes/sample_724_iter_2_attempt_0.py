import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Power output (hp)' to numeric, handling ranges like '200/235' by taking the second value
def parse_power(power_str):
    if '/' in power_str:
        return int(power_str.split('/')[1])
    return int(power_str)

df['Power output (hp)'] = df['Power output (hp)'].apply(parse_power)
# Sort by power output in descending order
ranked_models = df.sort_values(by='Power output (hp)', ascending=False)['Model']
print(f"Final Answer: {', '.join(ranked_models)}")