import pandas as pd

df_all_columns = pd.read_csv("base_diabetes_dataset.csv")

selected_columns = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']
df = df_all_columns[selected_columns]

for column in df.columns[:-1]: 
    mean_val = df[column].mean()
    std_dev = df[column].std()
    threshold = 3 * std_dev
    df.loc[~((mean_val - threshold <= df[column]) & (df[column] <= mean_val + threshold)), column] = float('nan')

df = df.dropna()


def group_values(df):
    df['Glucose'] = pd.cut(df['Glucose'], bins=[-float('inf'), 140, 199, float('inf')], labels=[1, 2, 3])
    df['BMI'] = pd.cut(df['BMI'], bins=[-float('inf'), 25, 30, float('inf')], labels=[1, 2, 3])
    df['DiabetesPedigreeFunction'] = pd.cut(df['DiabetesPedigreeFunction'], bins=[-float('inf'), 0.5, 1.5, 2, float('inf')],
                                             labels=[1, 2, 3, 4])
    df['Age'] = pd.cut(df['Age'], bins=[-float('inf'), 30, 45, 60, float('inf')], labels=[1, 2, 3, 4])

    return df

df = group_values(df)

df.to_csv("diabetes_dataset.csv", index=False) 

app_df = pd.read_csv("base_diabetes_app.csv")
app_df = group_values(df)
app_df.to_csv("diabetes_app.csv", index=False) 
