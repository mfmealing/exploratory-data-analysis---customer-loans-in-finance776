import yaml
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
from IPython.display import display


def load_yaml():
    with open('credentials.yaml', 'r') as f:
        credentials = yaml.safe_load(f)
    return credentials

class RDSDatabaseConnector:
    def __init__(self, creds):
        self.creds = creds

    def engine_init(self):
        engine = create_engine(f"postgresql+psycopg2://{self.creds['RDS_USER']}:{self.creds['RDS_PASSWORD']}@{self.creds['RDS_HOST']}:{self.creds['RDS_PORT']}/{self.creds['RDS_DATABASE']}")
        #engine.execution_options(isolation_level='AUTOCOMMIT').connect()
        return engine
    
    def create_df(self, engine):
        dataframe = pd.read_sql_table('loan_payments', engine)
        return dataframe
    
    def df_to_csv(self, dataframe):
        dataframe.to_csv('loan_payments.csv')

def csv_to_df():
    loans_df = pd.read_csv('loan_payments.csv', index_col=0)
    return loans_df


loans = RDSDatabaseConnector(load_yaml())
loans_engine = loans.engine_init()
loans_dataframe = loans.create_df(loans_engine)
loans.df_to_csv(loans_dataframe)

loans_df = csv_to_df()
print(loans_df.shape)
display(loans_df.head())