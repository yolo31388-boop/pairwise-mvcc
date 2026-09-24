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

版本可见性：每个事务在 begin() 时拿到单调递增的 tx_id 作为快照点；
版本以其提交事务的 tx_id 作为 commit_ts，仅当 commit_ts <= 快照点
时可见，因此旧快照天然看不到快照之后的提交。

写锁策略：
  - insert（key 在本事务快照中没有已提交版本）：保守加写锁，其他活跃
    事务对同一 key 的写 -> DeadlockError；
  - update（快照中已有已提交版本）：不加阻塞锁，commit 时做
    first-committer-wins 校验，版本过期 -> ConflictError。
"""
from __future__ import annotations

import json


class ConflictError(Exception):
    pass


class DeadlockError(Exception):
    pass


class MVCCStore:
    def __init__(self):
        # key -> list[(commit_ts, value)]，value is None 表示 tombstone
        self._versions: dict[str, list[tuple[int, object]]] = {}
        # tx_id -> {"snapshot": int, "writes": {key: (value, base_ts)}}
        self._txns: dict[int, dict] = {}
        # key -> 持有保守写锁的活跃事务 id
        self._locks: dict[str, int] = {}
        self._next_tx_id = 1

    # ---------------------------------------------------------- 事务
    def begin(self) -> int:
        tx_id = self._next_tx_id
        self._next_tx_id += 1
        self._txns[tx_id] = {"snapshot": tx_id, "writes": {}}
        return tx_id

    # ---------------------------------------------------------- 读
    def get(self, tx: int, key: str):
        state = self._txns[tx]
        # 先看本事务未提交的写（含 tombstone）
        if key in state["writes"]:
            return state["writes"][key][0]
        # 否则读快照点（含）之前最新的已提交版本
        for commit_ts, value in reversed(self._versions.get(key, ())):
            if commit_ts <= state["snapshot"]:
                return value
        return None

    def _visible_commit_ts(self, tx_id: int, key: str):
        """key 在 tx_id 快照下可见的最新已提交版本的 commit_ts，无则 None。"""
        for commit_ts, _ in reversed(self._versions.get(key, ())):
            if commit_ts <= tx_id:
                return commit_ts
        return None

    # ---------------------------------------------------------- 写
    def _write(self, tx: int, key: str, value) -> None:
        state = self._txns[tx]
        holder = self._locks.get(key)
        if holder is not None and holder != tx:
            # key 被其他活跃事务保守加锁（insert 场景）
            raise DeadlockError(f"write lock on {key!r} held by tx {holder}")
        base_ts = self._visible_commit_ts(state["snapshot"], key)
        state["writes"][key] = (value, base_ts)
        if base_ts is None:
            # 无已提交基线：insert，保守持锁直到提交/回滚
            self._locks[key] = tx

    def put(self, tx: int, key: str, value) -> None:
        self._write(tx, key, value)

    def delete(self, tx: int, key: str) -> None:
        self._write(tx, key, None)

    # ---------------------------------------------------------- 提交/回滚
    def commit(self, tx: int) -> None:
        state = self._txns.pop(tx)
        snapshot = state["snapshot"]
        # first-committer-wins：任一键在快照点之后出现了更高编号的提交
        for key in state["writes"]:
            latest = self._visible_commit_ts(self._next_tx_id - 1, key)
            if latest is not None and latest > snapshot:
                self._release_locks(tx, state)
                raise ConflictError(
                    f"key {key!r} committed by a newer tx after tx {tx} began"
                )
        # 原子可见：所有写以 tx 自身的 id 作为 commit_ts
        for key, (value, _base) in state["writes"].items():
            self._versions.setdefault(key, []).append((tx, value))
        self._release_locks(tx, state)

    def rollback(self, tx: int) -> None:
        state = self._txns.pop(tx)
        self._release_locks(tx, state)

    def _release_locks(self, tx: int, state: dict) -> None:
        locked = [k for k, owner in self._locks.items() if owner == tx]
        for key in locked:
            del self._locks[key]

    # ---------------------------------------------------------- WAL
    def persist_wal(self, path: str) -> None:
        # 只把已提交事务的写操作顺序写盘；活跃（未提交）事务不落盘。
        with open(path, "w", encoding="utf-8") as f:
            for key, chain in self._versions.items():
                for commit_ts, value in chain:
                    op = "delete" if value is None else "put"
                    f.write(
                        json.dumps(
                            {"tx": commit_ts, "key": key, "op": op, "value": value},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    def recover_from_wal(self, path: str) -> None:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # WAL 中只存在已提交记录（未提交事务从不写盘）；按 tx 分组重放，
        # 同一事务的多键写一起应用，保证原子性。
        by_tx: dict[int, list[dict]] = {}
        order: list[int] = []
        for rec in records:
            tx_id = rec["tx"]
            if tx_id not in by_tx:
                by_tx[tx_id] = []
                order.append(tx_id)
            by_tx[tx_id].append(rec)

        for tx_id in sorted(order):
            if tx_id not in self._txns:
                self._txns[tx_id] = {"snapshot": tx_id, "writes": {}}
            for rec in by_tx[tx_id]:
                if rec["op"] == "delete":
                    self.delete(tx_id, rec["key"])
                else:
                    self.put(tx_id, rec["key"], rec["value"])
            self.commit(tx_id)
        # 恢复后新事务的编号必须超过所有重放事务，否则快照点会过旧
        if order:
            self._next_tx_id = max(self._next_tx_id, max(order) + 1)
