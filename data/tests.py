
#%%
from sqlalchemy import create_engine
import	pandas as pd
import joblib

#%%
database_filepath = '../data/DisasterResponse.db'
engine = create_engine(f'sqlite:///{database_filepath}')

with engine.connect() as conn:
    df = pd.read_sql_table('disaster_table', conn)


# %%
df.head()

#%%
df.info()


# %%
df['genre'].value_counts()



# %%
model = joblib.load("../models/classifier.pkl")