"""MVCC 事务存储引擎（快照隔离 + 写冲突 + WAL 恢复）。

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

import json
import os

_TOMBSTONE = object()


class ConflictError(Exception):
    pass


class DeadlockError(Exception):
    pass


class MVCCStore:
    def __init__(self):
        # key -> [[version_txid, value], ...]，版本按提交顺序追加
        self._versions: dict[str, list[list]] = {}
        # txid -> {"snapshot": int, "writes": {key: value | _TOMBSTONE}}
        self._active: dict[int, dict] = {}
        self._next_tx_id = 1
        # 已提交事务日志：[(txid, ((key, value), ...)), ...]
        self._committed: list[tuple[int, tuple]] = []

    def begin(self) -> int:
        tx_id = self._next_tx_id
        self._next_tx_id += 1
        self._active[tx_id] = {"snapshot": tx_id, "writes": {}}
        return tx_id

    def get(self, tx: int, key: str):
        txn = self._active[tx]
        if key in txn["writes"]:
            value = txn["writes"][key]
            return None if value is _TOMBSTONE else value
        history = self._versions.get(key)
        if history:
            snapshot = txn["snapshot"]
            for version_tx_id, value in reversed(history):
                if version_tx_id <= snapshot:
                    return None if value is _TOMBSTONE else value
        return None

    def put(self, tx: int, key: str, value) -> None:
        self._write(tx, key, value)

    def delete(self, tx: int, key: str) -> None:
        self._write(tx, key, _TOMBSTONE)

    def _write(self, tx: int, key: str, value) -> None:
        txn = self._active[tx]
        writes = txn["writes"]
        if key not in writes:
            # 保守写锁检测：另一个活跃事务正持有着该 key 的未提交写。
            # 若该 key 从未被提交过（两个事务都在“插入”同一新 key），
            # 没有可在提交时做版本比较的基线，立即判死锁；
            # 已有已提交版本时放行，交给 commit 的 first-committer-wins 裁决。
            if key not in self._versions:
                for other_id, other_txn in self._active.items():
                    if other_id != tx and key in other_txn["writes"]:
                        raise DeadlockError(
                            f"key {key!r} is write-locked by tx {other_id}"
                        )
        writes[key] = value

    def commit(self, tx: int) -> None:
        # 先整体摘出事务，保证可见性是原子的（失败即回滚）。
        txn = self._active.pop(tx)
        writes = txn["writes"]
        snapshot = txn["snapshot"]
        for key in writes:
            history = self._versions.get(key)
            if history and history[-1][0] > snapshot:
                # 我 begin 之后已有更高编号事务提交了该 key -> 版本过期。
                raise ConflictError(
                    f"tx {tx} conflicts with committed version on key {key!r}"
                )
        for key, value in writes.items():
            self._versions.setdefault(key, []).append([tx, value])
        self._committed.append((tx, tuple(writes.items())))

    def rollback(self, tx: int) -> None:
        del self._active[tx]

    def persist_wal(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for tx_id, writes in self._committed:
                f.write(json.dumps({"tx": tx_id}) + "\n")
                for key, value in writes:
                    if value is _TOMBSTONE:
                        f.write(
                            json.dumps({"op": "del", "key": key}) + "\n"
                        )
                    else:
                        f.write(
                            json.dumps(
                                {"op": "put", "key": key, "value": value}
                            )
                            + "\n"
                        )
                f.write(json.dumps({"commit": tx_id}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def recover_from_wal(self, path: str) -> None:
        pending_tx = None
        pending_writes: dict[str, object] = {}
        try:
            f = open(path, "r", encoding="utf-8")
        except FileNotFoundError:
            return
        with f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "tx" in record:
                    pending_tx = record["tx"]
                    pending_writes = {}
                elif record.get("op") == "put":
                    pending_writes[record["key"]] = record["value"]
                elif record.get("op") == "del":
                    pending_writes[record["key"]] = _TOMBSTONE
                elif "commit" in record:
                    tx_id = record["commit"]
                    # 只重放完整提交的事务；文件尾部残缺事务被丢弃。
                    if tx_id == pending_tx:
                        self._replay_committed(tx_id, pending_writes)
                    pending_tx = None
                    pending_writes = {}

    def _replay_committed(self, tx_id: int, writes: dict) -> None:
        for key, value in writes.items():
            self._versions.setdefault(key, []).append([tx_id, value])
        self._committed.append((tx_id, tuple(writes.items())))
        self._next_tx_id = max(self._next_tx_id, tx_id + 1)
