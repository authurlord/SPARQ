import pandas as pd

df = pd.read_csv('table.csv')

# Define function to classify episode type
def get_episode_type(episode):
    if 'auditions' in episode:
        return 'auditions'
    elif 'semi-final' in episode:
        return 'semi-final'
    elif 'results' in episode:
        return 'results'
    else:
        return 'other'

# Apply the function to create a new column
df['episode_type'] = df['episode'].apply(get_episode_type)

# Group by episode type and compute correlation between 'official itv rating (millions)' and 'share (%)'
correlations = df.groupby('episode_type')[['official itv rating (millions)', 'share (%)']].corr()['share (%)']['official itv rating (millions)']

print(f"Final Answer: {correlations.to_dict()}")