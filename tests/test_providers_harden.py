"""
BaostockProvider 反扒硬化 / 磁盘缓存 / 错误必 raise / 线程安全 login —— 离线单测

全部 mock baostock（``src.data.providers.bs``），零网络、零真实 login。
覆盖 PRD 验收标准 A1-A3 / B4-B9 / C10-C13 / D14-D16 / E17-E20 / F21-F22。

运行：python -m pytest tests/test_providers_harden.py
（或：python tests/test_providers_harden.py）
"""

import ast
import logging
import os
import shutil
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src.data import providers as prov_mod
from src.data.providers import (
    BAOSTOCK_MIN_INTERVAL,
    BaostockProvider,
    CachedProvider,
    CSVProvider,
    DataProvider,
    RateLimiter,
    create_provider,
)
from src.core.cache import CacheConfig, FileCache, MemoryCache
from src.core.exceptions import DataFetchError

# BaostockProvider.API_FIELDS 的列顺序
FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
    "turn,tradestatus,pctChg,isST"
).split(",")


def make_row(date="2024-01-02", code="sh.600000", isST="0", tradestatus="1", close="10.5"):
    """构造一行 baostock 字符串行数据（顺序与 FIELDS 一致）。"""
    return [
        date, code, "10.0", "11.0", "9.0", close, "10.0",
        "1000", "10000", "3", "1.0", tradestatus, "0.5", isST,
    ]


class FakeRS:
    """模拟 baostock 结果集（next/get_row_data/error_code/error_msg/fields）。"""

    def __init__(self, error_code="0", error_msg="", rows=None, fields=None):
        self.error_code = error_code
        self.error_msg = error_msg
        self.fields = fields if fields is not None else FIELDS
        self._rows = rows or []
        self._i = -1

    def next(self):
        self._i += 1
        return self._i < len(self._rows)

    def get_row_data(self):
        return self._rows[self._i]


def _ok_rs(rows=None):
    """构造一个成功的 rs（含若干行）。"""
    return FakeRS(error_code="0", rows=rows if rows is not None else [make_row()])


def _query_returning(rows=None, error_code="0", error_msg=""):
    """side_effect 工厂：每次调用返回一个**全新**的 FakeRS（避免游标跨 load 调用复用）。"""
    def _fn(*a, **k):
        return FakeRS(
            error_code=error_code,
            error_msg=error_msg,
            rows=rows if rows is not None else [make_row()],
        )
    return _fn


def _no_cache():
    """禁用缓存的 CacheConfig（便于精确统计 baostock 调用次数）。"""
    return CacheConfig(enabled=False)


def _patched_bs(login_ok=True):
    """返回一个 patcher，patch ``src.data.providers.bs`` 并配置好 login。"""
    patcher = mock.patch("src.data.providers.bs")
    mock_bs = patcher.start()
    lg = mock.MagicMock()
    lg.error_code = "0" if login_ok else "1"
    lg.error_msg = "" if login_ok else "登录失败"
    mock_bs.login.return_value = lg
    return patcher, mock_bs


@pytest.fixture(autouse=True)
def _isolate_global_state():
    """每个测试前后：重置共享 login 状态；保存/恢复限速闸 min_interval。"""
    BaostockProvider._reset_login_state()
    saved = prov_mod._baostock_rate_limiter.min_interval
    # 默认关闭限速以加速非限速类测试；限速类测试自行设置 min_interval
    prov_mod._baostock_rate_limiter.min_interval = 0
    yield
    prov_mod._baostock_rate_limiter.min_interval = saved
    BaostockProvider._reset_login_state()
    mock.patch.stopall()


# ============================ A. 反扒限速下沉 + 统一保护 ============================

class TestRateLimiting:
    """A1-A3：最小间隔限速、多实例共享单例、可配置/可关。"""

    def test_a1_min_interval_throttle(self):
        """A1：连续调用 N 次，相邻 query 间隔 ≥ min_interval（容差内）。"""
        prov_mod._baostock_rate_limiter.min_interval = 0.1
        # 重置闸的“上次放行”时间戳，避免上一个测试残留影响
        prov_mod._baostock_rate_limiter._last_release = 0.0

        timestamps = []

        def _q(*a, **kw):
            timestamps.append(time.monotonic())
            return _ok_rs()

        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _q
        try:
            provider = BaostockProvider(cache_config=_no_cache())
            for _ in range(3):
                provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        finally:
            patcher.stop()

        assert len(timestamps) == 3
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        # 容差 30ms（调度抖动）
        for g in gaps:
            assert g >= 0.1 - 0.03, f"相邻请求间隔 {g:.4f}s 小于最小间隔（容差内）"

    def test_a2_shared_singleton_gate(self):
        """A2：两个独立 BaostockProvider 实例共享同一限速闸单例。"""
        p1 = BaostockProvider(cache_config=_no_cache())
        p2 = BaostockProvider(cache_config=_no_cache())
        # 直接断言实例引用的是模块级同一个 RateLimiter 对象
        assert p1._rate_limiter is p2._rate_limiter
        assert p1._rate_limiter is prov_mod._baostock_rate_limiter

    def test_a2_shared_gate_global_pacing(self):
        """A2（行为）：两实例交错调用，全局相邻间隔 ≥ min_interval。"""
        prov_mod._baostock_rate_limiter.min_interval = 0.1
        prov_mod._baostock_rate_limiter._last_release = 0.0

        timestamps = []

        def _q(*a, **kw):
            timestamps.append(time.monotonic())
            return _ok_rs()

        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _q
        try:
            p1 = BaostockProvider(cache_config=_no_cache())
            p2 = BaostockProvider(cache_config=_no_cache())
            # 交错：p1, p2, p1, p2 —— 若各自独立计时，跨实例间隔可能 < min_interval
            p1.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            p2.load("sz.000001", start_date="2024-01-01", end_date="2024-01-10")
            p1.load("sh.600000", start_date="2024-01-11", end_date="2024-01-20")
            p2.load("sz.000001", start_date="2024-01-11", end_date="2024-01-20")
        finally:
            patcher.stop()

        assert len(timestamps) == 4
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        for g in gaps:
            assert g >= 0.1 - 0.03, f"跨实例间隔 {g:.4f}s 说明未共享闸"

    def test_a3_disabled_when_zero(self):
        """A3：min_interval=0 退化为不限速（相邻间隔无下界）。"""
        prov_mod._baostock_rate_limiter.min_interval = 0
        prov_mod._baostock_rate_limiter._last_release = 0.0

        timestamps = []

        def _q(*a, **kw):
            timestamps.append(time.monotonic())
            return _ok_rs()

        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _q
        try:
            provider = BaostockProvider(cache_config=_no_cache())
            for _ in range(3):
                provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        finally:
            patcher.stop()

        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        # 不限速 → 间隔应非常小（远小于默认 0.2s）
        assert max(gaps) < 0.05

    def test_a3_custom_interval(self):
        """A3：min_interval=0.3 时相邻间隔 ≥ 0.3（容差内）。"""
        prov_mod._baostock_rate_limiter.min_interval = 0.3
        prov_mod._baostock_rate_limiter._last_release = 0.0

        timestamps = []

        def _q(*a, **kw):
            timestamps.append(time.monotonic())
            return _ok_rs()

        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _q
        try:
            provider = BaostockProvider(cache_config=_no_cache())
            for _ in range(2):
                provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        finally:
            patcher.stop()

        assert len(timestamps) == 2
        gap = timestamps[1] - timestamps[0]
        assert gap >= 0.3 - 0.03


# ============================ B. 礼貌重试 + 错误必 raise ============================

class TestRetryAndRaise:
    """B4-B9：重试后成功、重试耗尽必 raise、各类错误码语义。"""

    def test_b4_retry_then_success(self):
        """B4：前两次网络异常、第三次成功 → 返回 df，query 恰好 3 次。"""
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = [
            ConnectionError("connection timeout"),
            ConnectionError("connection timeout"),
            _ok_rs(),
        ]
        with mock.patch("src.data.providers.time.sleep"):  # 跳过退避等待
            try:
                provider = BaostockProvider(cache_config=_no_cache())
                df = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            finally:
                patcher.stop()

        assert df is not None
        assert len(df) == 1
        assert mock_bs.query_history_k_data_plus.call_count == 3

    def test_b5_exhaust_raises_network(self):
        """B5：恒抛网络异常 → 抛 DataFetchError(NETWORK_ERROR)，不返回 None。"""
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = ConnectionError("网络超时")
        with mock.patch("src.data.providers.time.sleep"):
            provider = BaostockProvider(cache_config=_no_cache())
            with pytest.raises(DataFetchError) as ei:
                provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            patcher.stop()

        err = ei.value
        assert err.error_code == DataFetchError.NETWORK_ERROR
        assert err.context.get("source") == "baostock"

    def test_b6_network_error_code_raises(self):
        """B6：rs.error_code != 0 且命中 network 关键词 → raise NETWORK_ERROR（修原 warning+None）。"""
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.return_value = FakeRS(
            error_code="1", error_msg="网络超时"
        )
        provider = BaostockProvider(cache_config=_no_cache())
        with pytest.raises(DataFetchError) as ei:
            provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        patcher.stop()

        assert ei.value.error_code == DataFetchError.NETWORK_ERROR

    def test_b7_unknown_error_code_raises(self):
        """B7：rs.error_code != 0 且不命中任何已知分类 → raise（不静默 None）。"""
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.return_value = FakeRS(
            error_code="1", error_msg="某种从未见过的错误 xyz123"
        )
        provider = BaostockProvider(cache_config=_no_cache())
        with pytest.raises(DataFetchError):
            provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        patcher.stop()

    def test_b8_data_error_returns_none(self):
        """B8：rs.error_code != 0 且命中 data 关键词 → 返回 None（合法空）。"""
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.return_value = FakeRS(
            error_code="1", error_msg="没有数据"
        )
        provider = BaostockProvider(cache_config=_no_cache())
        result = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        patcher.stop()

        assert result is None

    def test_b9_parameter_error_raises(self):
        """B9：参数错误 → raise（回归保护）。"""
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.return_value = FakeRS(
            error_code="1", error_msg="股票代码不存在"
        )
        provider = BaostockProvider(cache_config=_no_cache())
        with pytest.raises(DataFetchError) as ei:
            provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        patcher.stop()

        assert ei.value.error_code == DataFetchError.API_ERROR


# ============================ C. 磁盘缓存 ============================

class TestDiskCache:
    """C10-C13：首次落盘+命中免请求、TTL 过期刷新、写失败降级、直连路径享缓存。"""

    def test_c10_first_write_then_hit(self, tmp_path):
        """C10：CachedProvider(file) 首次落盘；同参第二次 mock query 计数不增，值等价。"""
        cache_dir = str(tmp_path / "cache")
        # 内层 BaostockProvider 关闭自身缓存，确保只有 CachedProvider 缓存生效
        inner = BaostockProvider(cache_config=_no_cache())
        provider = CachedProvider(inner, CacheConfig(backend="file", cache_dir=cache_dir))

        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _query_returning([make_row(), make_row(date="2024-01-03")])
        try:
            df1 = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            assert mock_bs.query_history_k_data_plus.call_count == 1
            # 缓存文件已生成，文件名/键含 code + start + end 组件
            files = os.listdir(cache_dir)
            assert len(files) == 1
            fname = files[0]
            assert "600000" in fname
            assert "start_date" in fname and "end_date" in fname

            # 同参第二次 → 走磁盘缓存，query 不再被调用
            df2 = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            assert mock_bs.query_history_k_data_plus.call_count == 1
        finally:
            patcher.stop()

        # 返回值等价（shape + 关键列值一致）
        assert df1 is not None and df2 is not None
        assert df1.shape == df2.shape
        assert list(df1["close"]) == list(df2["close"])

    def test_c11_ttl_expiry_refreshes(self, tmp_path):
        """C11：TTL 过期触发重新取数（query +1）；未过期则计数不变。"""
        cache_dir = str(tmp_path / "cache")
        inner = BaostockProvider(cache_config=_no_cache())
        provider = CachedProvider(
            inner, CacheConfig(backend="file", cache_dir=cache_dir, ttl=100)
        )

        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _query_returning()
        try:
            provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            assert mock_bs.query_history_k_data_plus.call_count == 1
            # 未过期：同参再 load，计数不变
            provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            assert mock_bs.query_history_k_data_plus.call_count == 1

            # 把缓存文件 mtime 调到 200s 前，模拟过期
            f = os.listdir(cache_dir)[0]
            old = time.time() - 200
            os.utime(os.path.join(cache_dir, f), (old, old))
            provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            assert mock_bs.query_history_k_data_plus.call_count == 2  # 重新取数
        finally:
            patcher.stop()

    def test_c12_write_failure_degrades_without_swallowing(self, tmp_path, caplog):
        """C12：cache_dir 只读 → 写失败降级，load 仍返回正确 df 并记 warning。"""
        ro_dir = str(tmp_path / "readonly")
        os.makedirs(ro_dir, exist_ok=True)
        os.chmod(ro_dir, 0o555)  # 只读

        try:
            mock_base = mock.MagicMock(spec=DataProvider)
            mock_base.load.return_value = pd.DataFrame({"date": [1], "close": [10.0]})
            provider = CachedProvider(
                mock_base, CacheConfig(backend="file", cache_dir=ro_dir)
            )
            with caplog.at_level(logging.WARNING):
                df = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")

            # 数据未被吞：返回正确 df
            assert df is not None
            assert len(df) == 1
            # 产生降级 warning（缓存是优化路径，写失败不抛）
            assert any("文件缓存写入失败" in r.message for r in caplog.records)
        finally:
            os.chmod(ro_dir, 0o755)  # 恢复以便清理

    def test_c13_default_baostock_uses_file_cache(self):
        """C13：BaostockProvider 默认经磁盘缓存（修 dead code）。"""
        provider = BaostockProvider()
        assert isinstance(provider._cache, FileCache)

    def test_c13_direct_path_cached(self, tmp_path):
        """C13（行为）：直连 BaostockProvider 同参重复 load 不重复打 baostock。"""
        cache_dir = str(tmp_path / "cache")
        provider = BaostockProvider(
            cache_config=CacheConfig(backend="file", cache_dir=cache_dir)
        )
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _query_returning()
        try:
            provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            assert mock_bs.query_history_k_data_plus.call_count == 1
            provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            assert mock_bs.query_history_k_data_plus.call_count == 1  # 命中缓存
        finally:
            patcher.stop()


# ============================ D. login 线程安全 ============================

class TestLoginThreadSafety:
    """D14-D16：共享单次 login、close 互不踩、login 失败必 raise。"""

    def test_d14_shared_single_login(self):
        """D14：两个实例分别 load，bs.login 全程仅 1 次。"""
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _query_returning()
        try:
            p1 = BaostockProvider(cache_config=_no_cache())
            p2 = BaostockProvider(cache_config=_no_cache())
            p1.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            p2.load("sz.000001", start_date="2024-01-01", end_date="2024-01-10")
        finally:
            patcher.stop()

        assert mock_bs.login.call_count == 1

    def test_d15_close_does_not_stomp_other(self):
        """D15：A.load 后 A.close，B（已登录）再 load 仍成功，且未触发全局 logout。"""
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _query_returning()
        try:
            p1 = BaostockProvider(cache_config=_no_cache())
            p2 = BaostockProvider(cache_config=_no_cache())
            p1.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
            p2.load("sz.000001", start_date="2024-01-01", end_date="2024-01-10")
            assert mock_bs.login.call_count == 1

            # A 关闭：此时 B 仍在登录态 → 不应 logout 全局会话
            p1.close()
            assert mock_bs.logout.call_count == 0

            # B 再 load：仍处于已登录态，成功返回
            df = p2.load("sz.000001", start_date="2024-01-11", end_date="2024-01-20")
            assert df is not None
            assert mock_bs.login.call_count == 1  # B 无需重新 login
        finally:
            patcher.stop()

    def test_d16_login_failure_raises(self):
        """D16：login 失败 → raise DataFetchError。"""
        patcher, _ = _patched_bs(login_ok=False)
        try:
            provider = BaostockProvider(cache_config=_no_cache())
            with pytest.raises(DataFetchError) as ei:
                provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        finally:
            patcher.stop()
        assert "baostock登录失败" in str(ei.value)


# ============================ E. 风格清理 ============================

class TestStyleCleanup:
    """E17-E20：& → and 语义不变、import 提顶、北交所正则、ST 过滤可选。"""

    def test_e17_collection_unchanged(self):
        """E17：& → and 后，多行 rs 的收集结果正确（行数/内容不变）。"""
        rows = [make_row(date=f"2024-01-0{i}") for i in range(1, 5)]
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _query_returning(rows)
        try:
            provider = BaostockProvider(cache_config=_no_cache())
            df = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        finally:
            patcher.stop()
        assert df is not None
        assert len(df) == 4  # 4 行全部收集
        assert list(df["date"].dt.strftime("%Y-%m-%d")) == [f"2024-01-0{i}" for i in range(1, 5)]

    def test_e18_top_level_imports(self):
        """E18：re/os 在模块顶部 import；方法内不再 import；ast 解析通过。"""
        src = open(
            os.path.join(os.path.dirname(prov_mod.__file__), "providers.py")
        ).read()
        tree = ast.parse(src)
        # 顶部 re / os 可用
        assert hasattr(prov_mod, "re"), "re 未在模块顶部导入"
        assert hasattr(prov_mod, "os"), "os 未在模块顶部导入"
        # 方法内不得再 import re / os
        in_method_imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Import):
                        for n in child.names:
                            in_method_imports.append(n.name)
                    elif isinstance(child, ast.ImportFrom):
                        in_method_imports.append(child.module)
        assert "re" not in in_method_imports, "方法内仍有 import re"
        assert "os" not in in_method_imports, "方法内仍有 import os"

    def test_e19_bj_regex(self):
        """E19：北交所正则 + 沪深用例回归不变。"""
        provider = BaostockProvider()
        # 北交所（新增）
        assert provider.exists("bj.830000") is True
        assert provider.exists("830000") is True
        # 沪深回归
        assert provider.exists("sh.600000") is True
        assert provider.exists("sz.000001") is True
        assert provider.exists("sz.300001") is True
        assert provider.exists("600000") is True
        assert provider.exists("000001") is True
        # 无效
        assert provider.exists("invalid") is False
        assert provider.exists("123456") is False

    def test_e20_filter_st_optional(self):
        """E20：filter_st=False 保留 ST/停牌行；默认 True 过滤掉。"""
        rows = [
            make_row(date="2024-01-02", isST="1", tradestatus="1"),  # ST 行
            make_row(date="2024-01-03", isST="0", tradestatus="1"),  # 正常行
        ]
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _query_returning(rows)
        try:
            provider = BaostockProvider(cache_config=_no_cache())
            # 默认过滤：ST 行被剔除，只剩 1 行
            df_filtered = provider.load(
                "sh.600000", start_date="2024-01-01", end_date="2024-01-10", filter_st=True
            )
            # 不过滤：保留全部 2 行（含 isST=1）
            df_all = provider.load(
                "sh.600000", start_date="2024-01-01", end_date="2024-01-10", filter_st=False
            )
        finally:
            patcher.stop()

        assert len(df_filtered) == 1
        assert set(df_filtered["isST"]) == {"0"}
        assert len(df_all) == 2
        assert "1" in set(df_all["isST"])


# ============================ F. 离线测试 / 构造签名兼容 ============================

class TestOfflineAndSignatures:
    """F21-F22：零网络、构造签名兼容既有调用点。"""

    def test_f21_all_mocked_no_network(self):
        """F21：所有 baostock 调用均被 mock（本文件全部测试即证；此处做一次端到端冒烟）。"""
        patcher, mock_bs = _patched_bs()
        mock_bs.query_history_k_data_plus.side_effect = _query_returning([make_row()])
        try:
            provider = BaostockProvider(cache_config=_no_cache())
            df = provider.load("sh.600000", start_date="2024-01-01", end_date="2024-01-10")
        finally:
            patcher.stop()
        assert df is not None
        # 真实 bs.login 未被调用（用的是 mock）
        assert mock_bs.login.called

    def test_f22_construction_signatures(self, tmp_path):
        """F22：既有调用点的构造签名均无需改动即可工作。"""
        # pipeline.py:285 / downloaders.py:413 风格
        p1 = BaostockProvider()
        p2 = BaostockProvider(config=None)
        assert isinstance(p1, BaostockProvider) and isinstance(p2, BaostockProvider)

        # batch_predictor.py:194 风格（CSVProvider）
        csv = CSVProvider(str(tmp_path))
        assert isinstance(csv, CSVProvider)

        # CachedProvider(provider, cache_config=None)
        cp = CachedProvider(p1, cache_config=None)
        assert isinstance(cp, CachedProvider)

        # create_provider 各类型
        assert isinstance(create_provider("baostock"), BaostockProvider)
        assert isinstance(create_provider("csv", base_dir=str(tmp_path)), CSVProvider)
        base = BaostockProvider()
        wrapped = create_provider("cached", provider=base)
        assert isinstance(wrapped, CachedProvider) and wrapped.provider is base

    def test_f22_ratelimiter_unit(self):
        """F22 辅助：RateLimiter 作为可复用单元，min_interval<=0 立即返回。"""
        rl = RateLimiter(min_interval=0)
        t0 = time.monotonic()
        rl.acquire()
        rl.acquire()
        assert time.monotonic() - t0 < 0.05  # 不限速，两次 acquire 几乎瞬时


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
