import pandas as pd

#Leitura do dataset
df_all_columns = pd.read_csv("base_diabetes_dataset.csv")
app_df = pd.read_csv("base_diabetes_app.csv")

#Seleção das colunas 
selected_columns = ['Pregnancies', 'Glucose','BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']
df = df_all_columns[selected_columns]

#Remoção dos outliers
for column in df.columns[:-1]: 
    mean_val = df[column].mean()
    std_dev = df[column].std()
    threshold = 3 * std_dev
    df.loc[~((mean_val - threshold <= df[column]) & (df[column] <= mean_val + threshold)), column] = float('nan')

#Tratamento dos NaNs
df = df.dropna()
#df = df.fillna(df.median())

#Normalização dos dados

def normalize_column(df, column, min_val, max_val, bias):
    df[column] = bias * ((df[column] - min_val) / (max_val - min_val))
    return df

min_bp = min(df['BloodPressure'].min(), app_df['BloodPressure'].min())
max_bp = max(df['BloodPressure'].max(), app_df['BloodPressure'].max())

df = normalize_column(df, 'BloodPressure', min_bp, max_bp, 1)
app_df = normalize_column(app_df, 'BloodPressure', min_bp, max_bp, 1)

min_st = min(df['SkinThickness'].min(), app_df['SkinThickness'].min())
max_st = max(df['SkinThickness'].max(), app_df['SkinThickness'].max())

df = normalize_column(df, 'SkinThickness', min_st, max_st, 2)
app_df = normalize_column(app_df, 'SkinThickness', min_st, max_st, 2)

correlation_matrix = df.corr()

#Salvar o dataset para o uso em diabetes_csv.py
app_df.to_csv("diabetes_app.csv", index=False)
df.to_csv("diabetes_dataset.csv", index=False)