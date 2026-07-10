"""
股票技术分析模块

对筛选出的股票进行技术分析，生成详细的分析报告
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)


class StockAnalyzer(AnalyzerBase):
    """
    股票技术分析器

    对选中的股票进行技术分析，生成交易信号和建议

    Example:
        >>> analyzer = StockAnalyzer()
        >>> result = analyzer.analyze_stocks(analysis_type="technical")
        >>> print(f"分析 {result['analyzed_count']} 只股票")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化股票分析器

        Args:
            config: 配置实例
        """
        super().__init__(config)
        self.reports_dir = self.config.data.reports_dir
        self.data_dir = self.config.data.data_dir

        # 确保报告目录存在
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)

    def load_selection_results(self) -> List[str]:
        """
        加载选股结果

        Returns:
            股票代码列表
        """
        selection_file = os.path.join(
            self.config.work_dir,
            self.config.data.selection_results_file
        )

        if not os.path.exists(selection_file):
            logger.warning(f"选股结果文件不存在: {selection_file}")
            return []

        try:
            with open(selection_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            selected_stocks = [
                item['code'] for item in data.get('selected_stocks', [])
            ]

            logger.info(f"加载 {len(selected_stocks)} 只选中股票")
            return selected_stocks

        except Exception as e:
            logger.error(f"加载选股结果失败: {e}")
            return []

    def _load_stock_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        加载股票数据

        Args:
            stock_code: 股票代码

        Returns:
            股票数据DataFrame，失败返回None
        """
        data_file = os.path.join(self.data_dir, f"{stock_code}.csv")

        if not os.path.exists(data_file):
            logger.warning(f"股票数据文件不存在: {data_file}")
            return None

        try:
            df = pd.read_csv(data_file)
            return df
        except Exception as e:
            logger.error(f"加载股票数据失败 {stock_code}: {e}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            df: 股票数据

        Returns:
            添加技术指标的DataFrame
        """
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        # MACD指标
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['DEA'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # KDJ指标
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        # RSI指标
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 布林带
        df['BOLL_MID'] = df['close'].rolling(window=20).mean()
        df['BOLL_STD'] = df['close'].rolling(window=20).std()
        df['BOLL_UPPER'] = df['BOLL_MID'] + 2 * df['BOLL_STD']
        df['BOLL_LOWER'] = df['BOLL_MID'] - 2 * df['BOLL_STD']

        return df

    def _generate_trading_signals(self, df: pd.DataFrame) -> List[str]:
        """
        生成交易信号

        Args:
            df: 包含技术指标的股票数据

        Returns:
            交易信号列表
        """
        signals = []

        if len(df) < 3:
            return ["数据不足，无法进行技术分析"]

        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3]

            buy_score = 0
            sell_score = 0

            # MACD信号分析
            if 'MACD' in df.columns and 'DEA' in df.columns:
                if latest['MACD'] > latest['DEA'] and prev['MACD'] <= prev['DEA']:
                    if latest['MACD'] > 0:
                        signals.append("🔥 MACD零轴上方金叉，强烈买入信号")
                        buy_score += 3
                    else:
                        signals.append("📈 MACD金叉，买入信号")
                        buy_score += 2
                elif latest['MACD'] < latest['DEA'] and prev['MACD'] >= prev['DEA']:
                    if latest['MACD'] < 0:
                        signals.append("🔻 MACD零轴下方死叉，强烈卖出信号")
                        sell_score += 3
                    else:
                        signals.append("📉 MACD死叉，卖出信号")
                        sell_score += 2

            # MACD柱状图分析
            if latest['MACD'] > prev['MACD'] > prev2['MACD']:
                signals.append("📊 MACD柱状图连续放大，动能增强")
                buy_score += 1

            # KDJ信号分析
            if 'K' in df.columns and 'D' in df.columns:
                if (latest['K'] > latest['D'] and prev['K'] <= prev['D'] and
                    latest['K'] < 30 and latest['D'] < 30):
                    signals.append("🚀 KDJ低位金叉，超跌反弹信号")
                    buy_score += 3
                elif (latest['K'] < latest['D'] and prev['K'] >= prev['D'] and
                      latest['K'] > 70 and latest['D'] > 70):
                    signals.append("⚠️ KDJ高位死叉，超买回调信号")
                    sell_score += 3

            # J值极值分析
            if 'J' in df.columns:
                if latest['J'] < 10:
                    signals.append("💎 J值极度超卖，关注抄底机会")
                    buy_score += 2
                elif latest['J'] > 90:
                    signals.append("🔥 J值极度超买，注意风险")
                    sell_score += 2

            # RSI分析
            if 'RSI' in df.columns:
                rsi_val = latest['RSI']
                if rsi_val < 20:
                    signals.append("💪 RSI严重超卖，强烈反弹信号")
                    buy_score += 3
                elif rsi_val < 30:
                    signals.append("📈 RSI超卖，关注买入机会")
                    buy_score += 2
                elif rsi_val > 80:
                    signals.append("⚡ RSI严重超买，强烈回调信号")
                    sell_score += 3
                elif rsi_val > 70:
                    signals.append("📉 RSI超买，注意风险")
                    sell_score += 2

            # 布林带分析
            if 'BOLL_UPPER' in df.columns and 'BOLL_LOWER' in df.columns:
                if latest['close'] <= latest['BOLL_LOWER']:
                    signals.append("🎯 价格触及布林带下轨，可能反弹")
                    buy_score += 2
                elif latest['close'] >= latest['BOLL_UPPER']:
                    signals.append("🎯 价格触及布林带上轨，可能回调")
                    sell_score += 2

            # 综合评分
            if buy_score >= 5:
                signals.append(f"🟢 综合买入评分: {buy_score} - 强烈推荐买入")
            elif buy_score >= 3:
                signals.append(f"🟡 综合买入评分: {buy_score} - 可以考虑买入")
            elif sell_score >= 5:
                signals.append(f"🔴 综合卖出评分: {sell_score} - 建议卖出")
            elif sell_score >= 3:
                signals.append(f"🟠 综合卖出评分: {sell_score} - 谨慎持有")

            return signals

        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return ["分析过程中出现错误"]

    def analyze_technical_indicators(self, stock_code: str) -> Dict[str, Any]:
        """
        分析技术指标

        Args:
            stock_code: 股票代码

        Returns:
            分析结果字典
        """
        df = self._load_stock_data(stock_code)
        if df is None or len(df) == 0:
            return {
                "stock_code": stock_code,
                "success": False,
                "error": "无法加载股票数据"
            }

        try:
            # 计算技术指标
            df = self._calculate_technical_indicators(df)

            # 获取最新数据
            latest = df.iloc[-1]

            # 生成交易信号
            signals = self._generate_trading_signals(df)

            return {
                "stock_code": stock_code,
                "success": True,
                "latest_price": float(latest['close']),
                "latest_date": str(latest['date']),
                "price_change_pct": float(latest.get('pctChg', 0)),
                "volume": int(latest.get('volume', 0)),
                "indicators": {
                    "MACD": float(latest.get('MACD', 0)),
                    "DEA": float(latest.get('DEA', 0)),
                    "K": float(latest.get('K', 50)),
                    "D": float(latest.get('D', 50)),
                    "J": float(latest.get('J', 50)),
                    "RSI": float(latest.get('RSI', 50)),
                },
                "signals": signals,
                "analysis_date": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"分析股票失败 {stock_code}: {e}")
            return {
                "stock_code": stock_code,
                "success": False,
                "error": str(e)
            }

    def generate_analysis_report(
        self,
        stock_code: str,
        analysis: Dict[str, Any]
    ) -> str:
        """
        生成分析报告

        Args:
            stock_code: 股票代码
            analysis: 分析结果

        Returns:
            报告文件路径
        """
        try:
            # 生成Markdown报告
            report_lines = [
                f"# {stock_code} 技术分析报告",
                "",
                f"**分析时间**: {analysis['analysis_date']}",
                f"**最新日期**: {analysis['latest_date']}",
                "",
                "## 基本信息",
                "",
                f"- 股票代码: {stock_code}",
                f"- 最新价格: {analysis['latest_price']:.2f}",
                f"- 涨跌幅: {analysis['price_change_pct']:.2f}%",
                f"- 成交量: {analysis['volume']:,}",
                "",
                "## 技术指标",
                "",
                f"- MACD: {analysis['indicators']['MACD']:.4f}",
                f"- DEA: {analysis['indicators']['DEA']:.4f}",
                f"- KDJ: K={analysis['indicators']['K']:.2f}, D={analysis['indicators']['D']:.2f}, J={analysis['indicators']['J']:.2f}",
                f"- RSI: {analysis['indicators']['RSI']:.2f}",
                "",
                "## 交易信号",
                "",
            ]

            for signal in analysis['signals']:
                report_lines.append(f"- {signal}")

            report_lines.append("")
            report_lines.append("---")
            report_lines.append("*本报告仅供参考，投资有风险，入市需谨慎*")

            report_content = "\n".join(report_lines)

            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{stock_code}_技术分析_{timestamp}.md"
            report_path = os.path.join(self.reports_dir, report_filename)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            logger.info(f"生成报告: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"生成报告失败 {stock_code}: {e}")
            return ""

    def analyze_stocks(
        self,
        analysis_type: str = "technical"
    ) -> Dict[str, Any]:
        """
        执行股票分析流程

        Args:
            analysis_type: 分析类型（目前仅支持 "technical"）

        Returns:
            执行结果字典
        """
        logger.info(f"开始股票分析: type={analysis_type}")

        # 加载选中的股票
        selected_stocks = self.load_selection_results()

        if not selected_stocks:
            logger.warning("没有选中的股票，跳过分析")
            return {
                "success": True,
                "analyzed_count": 0,
                "message": "没有选中的股票"
            }

        analyzed_count = 0
        failed_count = 0
        results = []

        for stock_code in selected_stocks:
            try:
                # 分析股票
                analysis = self.analyze_technical_indicators(stock_code)

                if analysis.get('success'):
                    # 生成报告
                    report_path = self.generate_analysis_report(stock_code, analysis)
                    analysis['report_path'] = report_path
                    results.append(analysis)
                    analyzed_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"分析股票失败 {stock_code}: {e}")
                failed_count += 1

        logger.info(f"股票分析完成: 成功 {analyzed_count}, 失败 {failed_count}")

        return {
            "success": True,
            "analyzed_count": analyzed_count,
            "failed_count": failed_count,
            "results": results,
            "message": f"分析完成 {analyzed_count} 只股票"
        }
