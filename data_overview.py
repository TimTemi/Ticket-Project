import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def summarize_dataframe(df):
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    categorical_cols = ['type', 'queue', 'priority', 'language']
    for col in categorical_cols:
        print(f"\nDistribution of {col}:\n", df[col].value_counts())

def plot_distributions(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='type', data=df, order=df['type'].value_counts().index)
    plt.title('Distribution of Ticket Types')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(y='queue', data=df, order=df['queue'].value_counts().index)
    plt.title('Distribution of Queues')
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x='priority', data=df, order=df['priority'].value_counts().index)
    plt.title('Distribution of Priorities')
    plt.show()

def preprocess_null_values(df):
    df['subject'].fillna('No Subject', inplace=True)
    for col in ['tag_5', 'tag_6', 'tag_7', 'tag_8']:
        df[col].fillna('Unknown', inplace=True)
    df.drop(columns=['tag_9'], inplace=True)  # Drop column with all null values
    return df

if __name__ == "__main__":
    df = load_data('./data/helpdesk_customer_tickets.csv')
    summarize_dataframe(df)
    plot_distributions(df)
    df = preprocess_null_values(df)
