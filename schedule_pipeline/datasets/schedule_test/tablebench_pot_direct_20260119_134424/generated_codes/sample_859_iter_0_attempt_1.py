import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'Total:' row for accurate analysis
df = df[df['Party'] != 'Total:']
# Convert '%' column to float by removing the % sign and converting to float
df['%'] = df['%'].str.replace('%', '').astype(float)
# Find the party with the highest percentage
max_party = df.loc[df['%'].idxmax(), 'Party']
max_percentage = df['%'].max()
# Get Forward Bloc (Ruikar) percentage
fb_ruikar_percentage = df[df['Party'] == 'Forward Bloc (Ruikar)']['%'].values[0]
# Calculate the difference
difference = max_percentage - fb_ruikar_percentage
print(f"Final Answer: Indian National Congress, {difference:.2f}")