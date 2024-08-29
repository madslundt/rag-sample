from typing import Iterator, Optional, Sequence
from langchain_core.stores import BaseStore
from sqlitedict import SqliteDict
from langchain_core.documents import Document

class Sqlitestore(BaseStore[str, Document]):
    db: SqliteDict
    def __init__(self, path: str, tablename: str):
        self.db = SqliteDict(path, tablename=tablename, autocommit=True)

    def mget(self, keys: list[str]) -> list[Document]:
        return [self.db[key] for key in keys]

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        for key, value in key_value_pairs:
            self.db[key] = value

    def mdelete(self, keys: Sequence[str]) -> None:
        for key in keys:
            if key in self.db:
                del self.db[key]

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        if prefix is None:
            yield from self.db.keys()
        else:
            for key in self.db.keys():
                if key.startswith(prefix):
                    yield key

def get_sqlitestore(path: str, tablename: str) -> Sqlitestore:
    return Sqlitestore(path, tablename)
