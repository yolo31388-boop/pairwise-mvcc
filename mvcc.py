"""MVCC 事务存储引擎骨架（快照隔离 + 写冲突 + WAL 恢复）。

模型：
  begin() -> tx_id    启动事务（快照点 = 当前已提交版本）
  get(tx, key)        读自己未提交的写，否则读 <= tx_id 的最新已提交版本
  put(tx, key, value) 写锁冲突（key 已被其他活跃事务写）-> DeadlockError
  delete(tx, key)     写 tombstone
  commit(tx)          写写冲突（自己读的版本已过期）-> ConflictError
  rollback(tx)        丢弃本事务未提交写
  persist_wal(path) / recover_from_wal(path)  已提交事务写盘/恢复

异常：
  ConflictError   提交时版本过期（first-committer-wins 失败方）
  DeadlockError   put 时活跃写锁冲突（保守检测）
"""
from __future__ import annotations


class ConflictError(Exception):
    pass


class DeadlockError(Exception):
    pass


class MVCCStore:
    def __init__(self):
        raise NotImplementedError

    def begin(self) -> int:
        raise NotImplementedError

    def get(self, tx: int, key: str):
        raise NotImplementedError

    def put(self, tx: int, key: str, value) -> None:
        raise NotImplementedError

    def delete(self, tx: int, key: str) -> None:
        raise NotImplementedError

    def commit(self, tx: int) -> None:
        raise NotImplementedError

    def rollback(self, tx: int) -> None:
        raise NotImplementedError

    def persist_wal(self, path: str) -> None:
        raise NotImplementedError

    def recover_from_wal(self, path: str) -> None:
        raise NotImplementedError
