import sqlite3
import os
from logger_config import logger

DB_PATH = "./kelly_database.db"

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except Exception as e:
        logger.error("Error connecting to database", exc_info=True)
        raise e

def initialize_database():
    """
    Initializes the database with necessary tables.
    """
    if not os.path.exists(DB_PATH):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Create a table for storing dataset info (for admin uploads)
            cursor.execute("""
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            # Create a table for rehearsal flags
            cursor.execute("""
            CREATE TABLE rehearsal_flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flag_name TEXT UNIQUE,
                flag_value INTEGER
            )
            """)
            conn.commit()
            conn.close()
            logger.info("Database initialized at %s", DB_PATH)
        except Exception as e:
            logger.error("Error initializing database", exc_info=True)
            raise e
    else:
        logger.info("Database already exists at %s", DB_PATH)

if __name__ == '__main__':
    initialize_database()