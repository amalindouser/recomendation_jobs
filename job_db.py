from sqlalchemy import create_engine
import pandas as pd

df = pd.read_csv("model/job_postings.csv")
engine = create_engine("sqlite:///job_postings.db")
df.to_sql("job_postings", engine, if_exists='replace', index=False)
