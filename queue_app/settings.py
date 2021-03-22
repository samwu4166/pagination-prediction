# settings.py
from dotenv import load_dotenv
load_dotenv()

import os
##Rabbit_MQ Setting
RABBIT_ACCOUNT = os.getenv("RABBIT_ACCOUNT")
RABBIT_PASSWORD = os.getenv("RABBIT_PASSWORD")
RABBIT_HOST = os.getenv("RABBIT_HOST")
RABBIT_PORT = os.getenv("RABBIT_PORT")
EVENT_QUEUE = os.getenv("EVENT_QUEUE")
TEST_WORKER_QUEUE = os.getenv("TEST_WORKER_QUEUE")
ETL_QUEUE = os.getenv("ETL_QUEUE")
PREDICT_ERROR_QUEUE = os.getenv("PREDICT_ERROR_QUEUE")
##MONGO Setting
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_PORT = os.getenv("MONGO_PORT")
DATABASE = os.getenv("DATABASE")
COLLECTION = os.getenv("COLLECTION")
