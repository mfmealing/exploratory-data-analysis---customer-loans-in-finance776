import yaml
from sqlalchemy import create_engine
import psycopg2
import pandas as pd

def load_yaml():
    with open('credentials.yaml', 'r') as f:
        credentials = yaml.safe_load(f)
    return credentials

class RDSDatabaseConnector:
    def __init__(self, creds):
        self.creds = creds

    def engine_init(self):
        engine = create_engine(f"postgresql+psycopg2://{self.creds['RDS_USER']}:{self.creds['RDS_PASSWORD']}@{self.creds['RDS_HOST']}:{self.creds['RDS_PORT']}/{self.creds['RDS_DATABASE']}")
        return engine
    
    def create_df(self, engine):
        dataframe = pd.read_sql_table('loan_payments', engine)
        return dataframe
    
    def df_to_csv(self, dataframe):
        dataframe.to_csv('loan_payments.csv')

if __name__ == "__main__":
    # Loads dataframe and converts to csv which is then saved
    loans = RDSDatabaseConnector(load_yaml())
    loans_engine = loans.engine_init()
    loans_dataframe = loans.create_df(loans_engine)
    loans.df_to_csv(loans_dataframe)