# database.py
import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///kelly_knowledge.db")
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Knowledge(Base):
    __tablename__ = "knowledge"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50))
    source = Column(String(255))
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    rehearsed = Column(Boolean, default=False)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Database initialization error: {e}")

def insert_knowledge(category: str, source: str, content: str):
    session = SessionLocal()
    try:
        item = Knowledge(category=category, source=source, content=content)
        session.add(item)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_relevant_knowledge(query: str, limit: int = 5, only_new: bool = True):
    session = SessionLocal()
    try:
        q = session.query(Knowledge).filter(Knowledge.content.ilike(f"%{query}%"))
        if only_new:
            q = q.filter(Knowledge.rehearsed == False)
        results = q.order_by(Knowledge.timestamp.desc()).limit(limit).all()
        return results
    except Exception as e:
        raise e
    finally:
        session.close()

def mark_as_rehearsed(ids):
    session = SessionLocal()
    try:
        items = session.query(Knowledge).filter(Knowledge.id.in_(ids)).all()
        for item in items:
            item.rehearsed = True
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

init_db()