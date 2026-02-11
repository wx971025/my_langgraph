from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import Optional

from utils import logger


class BaseMySQL:
    def __init__(self, 
        username: Optional[str] = None,
        passwd: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        *,
        db_name: Optional[str] = None,
    ) -> None:
        self.username = username or "root"
        self.passwd = passwd or "Wx19971025!"
        self.host = host or "172.29.0.6"
        self.port = port or 3306
        self.db_name = db_name

        if self.db_name:
            self.connect_to_db(db_name, auto_create=True)
        else:
            self._create_engine()

    def _get_uri(self, db_name: Optional[str] = None) -> str:
        if db_name:
            return f'mysql+pymysql://{self.username}:{self.passwd}@{self.host}:{self.port}/{db_name}?charset=utf8mb4'
        else:
            return f'mysql+pymysql://{self.username}:{self.passwd}@{self.host}:{self.port}?charset=utf8mb4'
    
    def _ensure_db_exists(self, db_name: Optional[str] = None):
        temp_uri = self._get_uri(db_name=None)
        temp_engine = create_engine(temp_uri)
        try:
            with temp_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
                conn.commit()
                logger.info(f"Database check/creation for '{db_name}' completed.")
        except Exception as e:
            logger.error(f"Failed to create database {db_name}: {e}")
            raise e
        finally:
            temp_engine.dispose()
    
    def _create_engine(self):
        if hasattr(self, 'engine'):
            self.engine.dispose()
        self.engine = create_engine(self._get_uri(self.db_name), echo=True)

    def connect_to_db(self, db_name: str, auto_create: bool = False) -> None:
        self.db_name = db_name
        if auto_create:
            self._ensure_db_exists(db_name)
        self._create_engine()

    def drop_db(self, db_name: str):
        temp_uri = self._get_uri(db_name=None)
        temp_engine = create_engine(temp_uri)
        try:
            with temp_engine.connect() as conn:
                conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
                conn.commit()
                logger.info(f"Database drop for '{db_name}' completed.")
        except Exception as e:
            logger.error(f"Failed to drop database {db_name}: {e}")
            raise e
        finally:
            temp_engine.dispose()



