"""Database abstraction layer to unify PostgreSQL and SQLite operations.

Provides a unified DB connection that automatically translates SQLite queries
to Postgres syntax (e.g. converting `?` to `%s`) if connected to Postgres.
"""

import sqlite3
import logging
from typing import Any, Tuple

from config import DATABASE_URL, DB_PATH

logger = logging.getLogger(__name__)


class CursorWrapper:
    def __init__(self, cursor, is_postgres: bool):
        self.cursor = cursor
        self.is_postgres = is_postgres

    def fetchone(self) -> Any:
        return self.cursor.fetchone()

    def fetchall(self) -> Any:
        return self.cursor.fetchall()

    @property
    def rowcount(self) -> int:
        return self.cursor.rowcount

    def __iter__(self):
        return iter(self.cursor)


class DBWrapper:
    """Wraps either a psycopg2 or sqlite3 connection."""
    
    def __init__(self, db_path: str = None):
        self.is_postgres = bool(DATABASE_URL)
        
        if self.is_postgres:
            import psycopg2
            from psycopg2.extras import DictCursor
            self.conn = psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor)
            self.conn.autocommit = True
        else:
            # Fall back to sqlite
            path = db_path if db_path else DB_PATH
            self.conn = sqlite3.connect(path)
            self.conn.row_factory = sqlite3.Row

    def _convert_sql(self, sql: str) -> str:
        """Naively converts SQLite syntax to PostgreSQL if needed."""
        if self.is_postgres:
            # Replace SQLite '?' parameters with Postgres '%s'
            sql = sql.replace("?", "%s")
            # Replace AUTOINCREMENT with SERIAL for table creation
            sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
            sql = sql.replace("AUTOINCREMENT", "SERIAL")
            # Replace DATETIME with TIMESTAMP
            sql = sql.replace("DATETIME", "TIMESTAMP")
            # SQLite handles INSERT OR IGNORE, Postgres uses ON CONFLICT DO NOTHING
            sql = sql.replace("INSERT OR IGNORE", "INSERT")
            if "INSERT" in sql and "ON CONFLICT" not in sql:
                # Naive fix for orders table primarily
                sql = sql.replace("VALUES (%s, %s, %s, %s)", "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING")
        return sql

    def execute(self, sql: str, params: Tuple = None) -> CursorWrapper:
        converted_sql = self._convert_sql(sql)
        cursor = self.conn.cursor()
        if params:
            cursor.execute(converted_sql, params)
        else:
            cursor.execute(converted_sql)
        return CursorWrapper(cursor, self.is_postgres)

    def executemany(self, sql: str, params_list: list) -> None:
        converted_sql = self._convert_sql(sql)
        cursor = self.conn.cursor()
        cursor.executemany(converted_sql, params_list)

    def executescript(self, sql: str) -> None:
        if self.is_postgres:
            # Postgres cursor.execute can handle multiple statements separated by ';'
            converted_sql = self._convert_sql(sql)
            cursor = self.conn.cursor()
            cursor.execute(converted_sql)
        else:
            self.conn.executescript(sql)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def get_db_connection(db_path: str = None) -> DBWrapper:
    """Get an abstracted database connection."""
    return DBWrapper(db_path)
