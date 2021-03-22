# settings.py
from dotenv import load_dotenv
load_dotenv()

import os
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_PORT = os.getenv("MONGO_PORT")
DATABASE = os.getenv("DATABASE")
COLLECTION = os.getenv("COLLECTION")