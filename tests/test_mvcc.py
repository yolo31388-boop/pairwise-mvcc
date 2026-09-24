"""MVCC 事务引擎验收测试：隔离/冲突/回滚/死锁/崩溃恢复。"""
import os

import pytest

from mvcc import ConflictError, DeadlockError, MVCCStore


# ------------------------------------------------------------ 基础
def test_basic_put_commit_get():
    s = MVCCStore()
    t = s.begin()
    s.put(t, "a", 1)
    s.commit(t)
    t2 = s.begin()
    assert s.get(t2, "a") == 1


def test_get_missing_none():
    s = MVCCStore()
    t = s.begin()
    assert s.get(t, "nope") is None


def test_rollback_discards():
    s = MVCCStore()
    t = s.begin()
    s.put(t, "a", 1)
    s.rollback(t)
    t2 = s.begin()
    assert s.get(t2, "a") is None


def test_update_value():
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "k", 1)
    s.commit(t1)
    t2 = s.begin()
    s.put(t2, "k", 2)
    s.commit(t2)
    t3 = s.begin()
    assert s.get(t3, "k") == 2


# ------------------------------------------------------------ 隔离
def test_uncommitted_invisible():
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "a", 1)  # 未提交
    t2 = s.begin()
    assert s.get(t2, "a") is None


def test_snapshot_does_not_see_later_commit():
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "a", 1)
    s.commit(t1)
    t_old = s.begin()          # 快照在 v1
    t2 = s.begin()
    s.put(t2, "a", 2)
    s.commit(t2)
    assert s.get(t_old, "a") == 1  # 旧快照仍读 v1
    t_new = s.begin()
    assert s.get(t_new, "a") == 2


# ------------------------------------------------------------ 写写冲突
def test_commit_conflict_detected():
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "a", 1)
    s.commit(t1)
    tx = s.begin()
    assert s.get(tx, "a") == 1
    s.put(tx, "a", 99)
    t2 = s.begin()
    s.put(t2, "a", 2)
    s.commit(t2)          # t2 先提交更新版
    with pytest.raises(ConflictError):
        s.commit(tx)      # tx 的版本已过期 -> 冲突
    t3 = s.begin()
    assert s.get(t3, "a") == 2


def test_same_key_write_lock_raises():
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "a", 1)
    t2 = s.begin()
    with pytest.raises(DeadlockError):
        s.put(t2, "a", 2)


def test_commit_after_rollback_no_leak():
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "a", 1)
    s.rollback(t1)
    t2 = s.begin()
    s.put(t2, "a", 2)
    s.commit(t2)
    t3 = s.begin()
    assert s.get(t3, "a") == 2


# ------------------------------------------------------------ 删除
def test_delete():
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "a", 1)
    s.commit(t1)
    t2 = s.begin()
    s.delete(t2, "a")
    s.commit(t2)
    t3 = s.begin()
    assert s.get(t3, "a") is None


def test_delete_then_reinsert():
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "a", 1)
    s.commit(t1)
    t2 = s.begin()
    s.delete(t2, "a")
    s.commit(t2)
    t3 = s.begin()
    s.put(t3, "a", 3)
    s.commit(t3)
    t4 = s.begin()
    assert s.get(t4, "a") == 3


# ------------------------------------------------------------ WAL 恢复
def test_wal_recover_committed(tmp_path):
    p = str(tmp_path / "wal.log")
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "a", 10)
    s.put(t1, "b", 20)
    s.commit(t1)
    t2 = s.begin()
    s.put(t2, "a", 99)
    s.rollback(t2)         # 未提交，不应恢复
    s.persist_wal(p)
    s2 = MVCCStore()
    s2.recover_from_wal(p)
    t = s2.begin()
    assert s2.get(t, "a") == 10
    assert s2.get(t, "b") == 20


def test_wal_recover_deletes(tmp_path):
    p = str(tmp_path / "wal2.log")
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "a", 1)
    s.commit(t1)
    t2 = s.begin()
    s.delete(t2, "a")
    s.commit(t2)
    s.persist_wal(p)
    s2 = MVCCStore()
    s2.recover_from_wal(p)
    t = s2.begin()
    assert s2.get(t, "a") is None


def test_recover_from_empty_wal(tmp_path):
    p = str(tmp_path / "wal3.log")
    s = MVCCStore()
    s.persist_wal(p)
    s2 = MVCCStore()
    s2.recover_from_wal(p)
    t = s2.begin()
    assert s2.get(t, "x") is None
