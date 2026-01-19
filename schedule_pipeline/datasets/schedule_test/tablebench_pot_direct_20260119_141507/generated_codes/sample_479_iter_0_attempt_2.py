import pandas as pd

df = pd.read_csv('table.csv')

# Identify rows where 'Apps' is negative or contains question marks
anomalies = df[(df['Apps'].astype(str).str.contains(r'[-]?[0-9]+[\?\-]?', na=False)) | 
               (df['Apps'].astype(str).str.contains(r'-', na=False))]

# Filter for negative Apps or suspicious values
negative_apps = df[df['Apps'] < 0]
question_mark_entries = df[df['Apps'].astype(str).str.contains(r'\?', na=False)]

# Combine and show anomalies
anomalies_list = []
if not negative_apps.empty:
    anomalies_list.append("Negative Apps: " + str(negative_apps[['Club', 'Season', 'Apps']].to_dict('records')))
if not question_mark_entries.empty:
    anomalies_list.append("Question marks in Apps: " + str(question_mark_entries[['Club', 'Season', 'Apps']].to_dict('records')))

# Also check for extremely high or low goals
outliers_goals = df[(df['Goals'] > 100) | (df['Goals'] < 0)]
if not outliers_goals.empty:
    anomalies_list.append("Extreme Goals: " + str(outliers_goals[['Club', 'Season', 'Goals']].to_dict('records')))

# Print the final list of anomalies
print(f"Final Answer: {anomalies_list[0]}")