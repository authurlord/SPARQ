import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Power output (hp)' to numeric, handling cases like '200/235'
def extract_max_power(power_str):
    if '/' in power_str:
        return max(map(int, power_str.split('/')))
    return int(power_str)

df['Power output (hp)'] = df['Power output (hp)'].apply(extract_max_power)

# Sort by power output in descending order
ranked_models = df.sort_values(by='Power output (hp)', ascending=False)['Model']

# Output the result
print(f"Final Answer: {', '.join(ranked_models)}")