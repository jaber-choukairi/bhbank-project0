from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Remplace user:pwd@localhost/dbname par ta configuration MySQL
DATABASE_URL = "mysql+pymysql://root:Jqber123***@127.0.0.1:3306/bhbank"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
