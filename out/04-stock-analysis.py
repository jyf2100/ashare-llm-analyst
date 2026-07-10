import base64
import os
import json
from datetime import datetime, timedelta
from io import BytesIO
from string import Template
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import pytz
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import Ashare as as_api
import MyTT_optimized as mt  # 使用优化后的技术指标库
from llm import LLMAnalyzer
import baostock as bs
import pandas as pd
import time
import logging
import shutil
import glob
# from baostock_rps_collector import BaostockRPSCollector  # 不再需要，改为从文件读取RPS数据

# 加载 .env 文件中的环境变量
load_dotenv()


def generate_trading_signals(df):
    """生成交易信号和建议 - 使用优化后的多指标综合分析"""
    signals = []
    
    # 检查数据是否足够进行分析
    if len(df) < 3:
        return ["数据不足，无法进行技术分析"]

    try:
        # 信号强度评分系统
        buy_score = 0
        sell_score = 0
        
        # 获取最新数据和历史数据
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # 1. MACD信号分析 - 增强版
        dif_col = 'DIF' if 'DIF' in df.columns else 'DIF_DMA'
        dea_col = 'DEA' if 'DEA' in df.columns else 'DIFMA_DMA'
        
        if dif_col in df.columns and dea_col in df.columns:
            if latest[dif_col] > latest[dea_col] and prev[dif_col] <= prev[dea_col]:
                if latest['MACD'] > 0:  # 零轴上方金叉更强
                    signals.append("🔥 MACD零轴上方金叉，强烈买入信号")
                    buy_score += 3
                else:
                    signals.append("📈 MACD金叉，买入信号")
                    buy_score += 2
            elif latest[dif_col] < latest[dea_col] and prev[dif_col] >= prev[dea_col]:
                if latest['MACD'] < 0:  # 零轴下方死叉更弱
                    signals.append("🔻 MACD零轴下方死叉，强烈卖出信号")
                    sell_score += 3
                else:
                    signals.append("📉 MACD死叉，卖出信号")
                    sell_score += 2
        
        # MACD柱状图分析
        if latest['MACD'] > prev['MACD'] > prev2['MACD']:
            signals.append("📊 MACD柱状图连续放大，动能增强")
            buy_score += 1

        # 2. KDJ信号分析 - 增强版
        # 低位金叉
        if (latest['K'] > latest['D'] and prev['K'] <= prev['D'] and 
            latest['K'] < 30 and latest['D'] < 30):
            signals.append("🚀 KDJ低位金叉，超跌反弹信号")
            buy_score += 3
        # 高位死叉
        elif (latest['K'] < latest['D'] and prev['K'] >= prev['D'] and 
              latest['K'] > 70 and latest['D'] > 70):
            signals.append("⚠️ KDJ高位死叉，超买回调信号")
            sell_score += 3
        
        # J值极值分析
        if latest['J'] < 10:
            signals.append("💎 J值极度超卖，关注抄底机会")
            buy_score += 2
        elif latest['J'] > 90:
            signals.append("🔥 J值极度超买，注意风险")
            sell_score += 2

        # 3. RSI多周期分析
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
            signals.append("📉 RSI超买，关注卖出机会")
            sell_score += 2

        # 4. 布林带突破分析 - 增强版
        close_price = latest['close']
        prev_close = prev['close']
        
        # 突破上轨
        if close_price > latest['BOLL_UPPER'] and prev_close <= prev['BOLL_UPPER']:
            signals.append("🚀 突破布林带上轨，强势上涨")
            buy_score += 2
        # 跌破下轨
        elif close_price < latest['BOLL_LOWER'] and prev_close >= prev['BOLL_LOWER']:
            signals.append("💎 跌破布林带下轨，超跌反弹机会")
            buy_score += 2
        # 回归中轨
        elif (abs(close_price - latest['BOLL_MID']) / latest['BOLL_MID'] < 0.01):
            signals.append("⚖️ 价格回归布林带中轨，趋势可能转换")

        # 5. DMI趋势强度分析
        adx_val = latest['ADX']
        if latest['PDI'] > latest['MDI'] and adx_val > 30:
            signals.append(f"📈 DMI多头趋势强劲 (ADX: {adx_val:.1f})")
            buy_score += 2
        elif latest['MDI'] > latest['PDI'] and adx_val > 30:
            signals.append(f"📉 DMI空头趋势强劲 (ADX: {adx_val:.1f})")
            sell_score += 2
        elif adx_val < 20:
            signals.append("😴 ADX显示趋势较弱，市场盘整")

        # 6. 成交量分析 - 新增OBV
        if 'OBV' in df.columns:
            if latest['OBV'] > prev['OBV'] > prev2['OBV']:
                signals.append("📊 OBV连续上升，资金持续流入")
                buy_score += 1
            elif latest['OBV'] < prev['OBV'] < prev2['OBV']:
                signals.append("📊 OBV连续下降，资金持续流出")
                sell_score += 1

        # 7. VR成交量比率分析
        vr_val = latest['VR']
        if vr_val > 400:
            signals.append(f"🔥 VR过高 ({vr_val:.1f})，市场过热风险")
            sell_score += 2
        elif vr_val < 50:
            signals.append(f"💎 VR过低 ({vr_val:.1f})，市场低迷机会")
            buy_score += 2
        elif 80 <= vr_val <= 120:
            signals.append(f"⚖️ VR正常范围 ({vr_val:.1f})，市场健康")

        # 8. ROC变动率分析
        if latest['ROC'] > latest['MAROC'] and prev['ROC'] <= prev['MAROC']:
            signals.append("🚀 ROC上穿均线，动能转强")
            buy_score += 2
        elif latest['ROC'] < latest['MAROC'] and prev['ROC'] >= prev['MAROC']:
            signals.append("📉 ROC下穿均线，动能转弱")
            sell_score += 2

        # 9. 多空指标BBI分析 - 新增
        if 'BBI' in df.columns:
            if latest['close'] > latest['BBI'] and prev['close'] <= prev['BBI']:
                signals.append("📈 价格突破多空线，多头占优")
                buy_score += 1
            elif latest['close'] < latest['BBI'] and prev['close'] >= prev['BBI']:
                signals.append("📉 价格跌破多空线，空头占优")
                sell_score += 1

        # 10. 综合信号评估
        net_score = buy_score - sell_score
        if net_score >= 5:
            signals.insert(0, "🔥🔥🔥 综合评估：强烈买入信号！")
        elif net_score >= 3:
            signals.insert(0, "🔥🔥 综合评估：买入信号")
        elif net_score >= 1:
            signals.insert(0, "📈 综合评估：偏多信号")
        elif net_score <= -5:
            signals.insert(0, "🔻🔻🔻 综合评估：强烈卖出信号！")
        elif net_score <= -3:
            signals.insert(0, "🔻🔻 综合评估：卖出信号")
        elif net_score <= -1:
            signals.insert(0, "📉 综合评估：偏空信号")
        else:
            signals.insert(0, "⚖️ 综合评估：震荡整理，观望为主")
        
        # 添加评分详情
        signals.append(f"📊 信号评分：买入 {buy_score} 分，卖出 {sell_score} 分，净值 {net_score} 分")

    except Exception as e:
        print(f"生成交易信号时出错: {str(e)}")
        signals.append(f"技术分析计算出错: {str(e)}")

    return signals if signals else ["当前无明显交易信号"]


def plot_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return image_base64


def _get_value_class(value):
    """根据数值返回CSS类名"""
    try:
        if isinstance(value, str) and '%' in value:
            value = float(value.strip('%'))
        elif isinstance(value, str):
            return 'neutral'
        if value > 0:
            return 'positive'
        elif value < 0:
            return 'negative'
        return 'neutral'
    except (ValueError, TypeError) as e:
        print(f"无法解析数值 {value}，错误信息: {e}")
        return 'neutral'


def _generate_table_row(key, value):
    """生成表格行HTML，包含样式"""
    value_class = _get_value_class(value)
    return f'<tr><td>{key}</td><td class="{value_class}">{value}</td></tr>'


class StockAnalyzer:
    def __init__(self, _stock_info, count=120, llm_api_key=None, llm_base_url=None, llm_model=None):
        """
        初始化股票分析器

        Args:
            _stock_info: 股票信息字典
            count: 获取的数据条数
            llm_api_key: llm API密钥，默认从环境变量LLM_API_KEY获取
            llm_base_url: llm API基础URL，默认从环境变量LLM_BASE_URL获取
            llm_model: llm 模型名称，默认从环境变量LLM_MODEL获取
        """
        self.stock_codes = list(_stock_info.values())
        self.stock_names = _stock_info
        self.count = count
        self.data = {}
        self.rps_data = {}  # 存储RPS数据
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 从环境变量获取API密钥和基础URL
        self.llm_api_key = llm_api_key or os.environ.get('LLM_API_KEY')
        self.llm_base_url = llm_base_url or os.environ.get('LLM_BASE_URL')
        self.llm_model = llm_model or os.environ.get('LLM_MODEL')

        # 初始化llm分析器
        self.llm = LLMAnalyzer(self.llm_api_key, self.llm_base_url, self.llm_model) if self.llm_api_key else None
        
        # 不再需要RPS采集器，改为从文件读取RPS数据
        # self.rps_collector = BaostockRPSCollector()
        
        # 配置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def backup_old_md_files(self, public_dir='public'):
        """备份public目录下非当日的md文件"""
        try:
            current_date = datetime.now().strftime('%Y%m%d')
            
            # 查找所有md文件
            md_files = glob.glob(os.path.join(public_dir, '*.md'))
            
            if not md_files:
                self.logger.info("未找到需要备份的md文件")
                return
            
            # 筛选出非当日的md文件
            old_files = []
            for file_path in md_files:
                filename = os.path.basename(file_path)
                # 检查文件名是否包含当前日期
                if current_date not in filename:
                    old_files.append(file_path)
            
            if not old_files:
                self.logger.info("未找到需要备份的非当日md文件")
                return
            
            # 创建备份目录
            backup_base_dir = 'back'
            if not os.path.exists(backup_base_dir):
                os.makedirs(backup_base_dir)
            
            # 创建带时间戳的备份目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = os.path.join(backup_base_dir, f'public_backup_{timestamp}')
            os.makedirs(backup_dir)
            
            # 移动非当日的md文件到备份目录
            moved_count = 0
            for file_path in old_files:
                filename = os.path.basename(file_path)
                backup_path = os.path.join(backup_dir, filename)
                shutil.move(file_path, backup_path)
                moved_count += 1
                self.logger.info(f"已备份文件: {filename} -> {backup_path}")
            
            self.logger.info(f"成功备份 {moved_count} 个非当日md文件到: {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"备份md文件时出错: {str(e)}")
    
    def save_raw_analysis_to_file(self, code, raw_analysis_text):
        """保存大模型原始输出到文件"""
        try:
            # 确保public目录存在
            public_dir = 'public'
            if not os.path.exists(public_dir):
                os.makedirs(public_dir)
            
            # 在生成新文件前，先备份非当日的md文件
            self.backup_old_md_files(public_dir)
            
            # 生成文件名：{code}_{yyyymmdd}.md
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f"{code}_{current_date}.md"
            filepath = os.path.join(public_dir, filename)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {self.get_stock_name(code)} ({code}) AI分析报告\n\n")
                f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## 大模型原始输出\n\n")
                f.write(raw_analysis_text)
            
            print(f"原始分析内容已保存到: {filepath}")
            return filepath
        except Exception as e:
            print(f"保存原始分析内容失败: {str(e)}")
            return None

    def get_stock_name(self, code):
        """根据股票代码获取股票名称"""
        return {v: k for k, v in self.stock_names.items()}.get(code, code)

    def fetch_data(self):
        """获取股票数据"""
        for code in self.stock_codes:
            stock_name = self.get_stock_name(code)
            try:
                print(f"正在获取股票 {stock_name} ({code}) 的数据...")
                df = as_api.get_price(code, count=self.count, frequency='1d')

                # 检查数据是否有效
                if df is None or df.empty:
                    print(f"警告：股票 {stock_name} ({code}) 返回空数据")
                    print(f"请检查股票代码是否正确。常见格式:")
                    print(f"  - 上交所: sh000001 (上证指数), sh600000 (浦发银行)")
                    print(f"  - 深交所: sz399001 (深证成指), sz000001 (平安银行)")
                    continue

                print(f"成功获取 {stock_name} ({code}) 数据，共 {len(df)} 条记录")
                print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
                self.data[code] = df

            except Exception as e:
                print(f"获取股票 {stock_name} ({code}) 数据失败: {str(e)}")
                print(f"建议检查:")
                print(f"  1. 股票代码格式是否正确 (如: sz002640 而不是 sh002640)")
                print(f"  2. 网络连接是否正常")
                print(f"  3. 股票是否已停牌或退市")
        
        # 获取RPS数据
        print("正在获取RPS数据...")
        self.fetch_rps_data()
    
    def fetch_rps_data(self):
        """
        从rps_results目录中读取RPS数据
        """
        try:
            self.logger.info("开始从rps_results目录读取RPS数据")
            
            # RPS数据目录
            rps_dir = "/mnt/disk01/workspaces/worksummary/ashare-llm-analyst/out/rps_results/"
            
            # 查找最新的RPS数据文件
            rps_files = []
            for file in os.listdir(rps_dir):
                if file.startswith('RPS') and file.endswith('.csv'):
                    rps_files.append(file)
            
            if not rps_files:
                self.logger.error("未找到RPS数据文件")
                self._use_default_rps()
                return
            
            # 按日期排序，获取最新的文件
            rps_files.sort(reverse=True)
            
            # 初始化数据存储
            self.rps_data = {}
            self.latest_rps = {}
            
            # 读取不同周期的RPS数据
            rps_periods = [5, 10, 20, 60, 120, 250]
            
            for period in rps_periods:
                period_files = [f for f in rps_files if f.startswith(f'RPS{period}_')]
                if period_files:
                    latest_file = period_files[0]  # 最新的文件
                    file_path = os.path.join(rps_dir, latest_file)
                    
                    try:
                        # 读取CSV文件
                        df = pd.read_csv(file_path)
                        self.logger.info(f"成功读取{period}日RPS数据文件: {latest_file}，共{len(df)}条记录")
                        
                        # 处理数据格式
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # 为当前分析的股票提取RPS数据
                        for code in self.stock_codes:
                            # 转换股票代码格式（如果需要）
                            baostock_code = self._convert_to_baostock_code(code)
                            
                            # 查找该股票的RPS数据
                            stock_data = df[df['stock_code'] == baostock_code].copy()
                            
                            if not stock_data.empty:
                                # 设置日期为索引
                                stock_data = stock_data.set_index('date')
                                
                                # 存储时间序列数据（使用20日RPS作为主要数据）
                                if period == 20:
                                    self.rps_data[code] = stock_data['rps']
                                    # 获取最新RPS值
                                    if not stock_data.empty:
                                        self.latest_rps[code] = stock_data['rps'].iloc[-1]
                                        self.logger.info(f"股票 {code} 最新RPS值: {self.latest_rps[code]:.2f}")
                                
                                # 存储多周期RPS数据（用于更详细的分析）
                                if code not in self.rps_data:
                                    self.rps_data[code] = {}
                                if not isinstance(self.rps_data[code], dict):
                                    # 如果已经是Series，转换为dict格式
                                    main_rps = self.rps_data[code]
                                    self.rps_data[code] = {'main': main_rps}
                                
                                self.rps_data[code][f'rps_{period}'] = stock_data['rps']
                            else:
                                self.logger.warning(f"未找到股票 {code} (转换为{baostock_code}) 的{period}日RPS数据")
                    
                    except Exception as e:
                        self.logger.error(f"读取{period}日RPS数据文件失败: {str(e)}")
            
            # 检查是否有股票缺少RPS数据
            missing_stocks = []
            for code in self.stock_codes:
                if code not in self.latest_rps:
                    missing_stocks.append(code)
                    # 使用默认值
                    self.latest_rps[code] = 50.0
                    self.rps_data[code] = pd.Series([50.0], index=[datetime.now()])
            
            if missing_stocks:
                self.logger.warning(f"以下股票未找到RPS数据，使用默认值: {missing_stocks}")
            
            self.logger.info(f"RPS数据读取完成，共处理 {len(self.latest_rps)} 只股票")
                    
        except Exception as e:
            self.logger.error(f"读取RPS数据时出错: {str(e)}")
            self._use_default_rps()
    
    def _use_default_rps(self):
        """使用默认RPS值"""
        self.rps_data = {}
        self.latest_rps = {}
        for code in self.stock_codes:
            self.rps_data[code] = pd.Series([50.0], index=[datetime.now()])
            self.latest_rps[code] = 50.0
        self.logger.info("使用默认RPS值 (50.0)")
    
    def _convert_to_baostock_code(self, code):
        """
        将8位股票代码转换为9位Baostock格式
        
        Args:
            code: 8位股票代码，如'sz002487'或'sh603063'
            
        Returns:
            str: 9位Baostock格式代码，如'sz.002487'或'sh.603063'
        """
        if '.' in code:
            return code  # 已经是9位格式
        
        if code.startswith('sz') or code.startswith('sh'):
            market = code[:2]
            stock_num = code[2:]
            return f"{market}.{stock_num}"
        else:
            # 默认处理：如果是6位数字，根据首位判断市场
            if len(code) == 6 and code.isdigit():
                if code.startswith('0') or code.startswith('3'):
                    return f"sz.{code}"
                elif code.startswith('6'):
                    return f"sh.{code}"
        
        return code  # 无法转换时返回原代码
    
    # def _fetch_missing_rps_data(self, missing_stocks, existing_latest_rps):
    #     """
    #     为缺失的股票单独获取数据并计算RPS
    #     注意：此方法已废弃，现在从rps_results目录读取RPS数据
    #     
    #     Args:
    #         missing_stocks: 缺失RPS数据的股票代码列表
    #         existing_latest_rps: 已有的最新RPS数据字典
    #     """
    #     # 此方法已废弃，改为从文件读取RPS数据
    #     pass
    

    
    def get_latest_rps(self, code):
        """
        获取指定股票的最新RPS值
        """
        # 优先使用缓存的最新RPS值
        if hasattr(self, 'latest_rps') and code in self.latest_rps:
            return self.latest_rps[code]
        
        # 备选方案：从时间序列数据中获取最新值
        if code in self.rps_data and len(self.rps_data[code]) > 0:
            return self.rps_data[code].iloc[-1]
        
        # 默认返回50.0（中性值）
        return 50.0

    def calculate_indicators(self, code):
        """计算技术指标 - 使用优化后的算法"""
        if code not in self.data:
            print(f"错误: 股票代码 {code} 没有数据")
            return None

        df = self.data[code].copy()

        # 检查数据量是否足够计算技术指标
        if len(df) < 60:  # 至少需要60天数据来计算各种指标
            print(f"警告: 股票 {code} 数据量不足 ({len(df)} 条)，可能影响技术指标计算准确性")

        try:
            # 使用优化后的数据类型转换，提升性能
            close = np.asarray(df['close'], dtype=np.float64)
            open_price = np.asarray(df['open'], dtype=np.float64)
            high = np.asarray(df['high'], dtype=np.float64)
            low = np.asarray(df['low'], dtype=np.float64)
            volume = np.asarray(df['volume'], dtype=np.float64)

            # 使用优化后的基础指标计算
            print(f"正在计算 {self.get_stock_name(code)} 的技术指标...")
            
            # MACD指标 - 使用优化算法
            dif, dea, macd = mt.MACD(close, SHORT=12, LONG=26, M=9)
            
            # KDJ随机指标 - 增强稳定性
            k, d, j = mt.KDJ(close, high, low, N=9, M1=3, M2=3)
            
            # 布林带 - 优化计算精度
            upper, mid, lower = mt.BOLL(close, N=20, P=2)
            
            # RSI相对强弱指标 - 增加异常处理
            rsi = mt.RSI(close, N=14)
            rsi = np.nan_to_num(rsi, nan=50)  # 处理NaN值
            
            # 心理线指标 - 改进计算逻辑
            psy, psyma = mt.PSY(close, N=12, M=6)
            
            # 威廉指标 - 增加除零保护
            wr, wr1 = mt.WR(close, high, low, N=10, N1=6)
            
            # 乖离率 - 增加除零保护
            bias1, bias2, bias3 = mt.BIAS(close, L1=6, L2=12, L3=24)
            
            # 顺势指标 - 增加除零保护
            cci = mt.CCI(close, high, low, N=14)

            # 移动平均线 - 使用向量化优化算法
            ma5 = mt.MA(close, 5)
            ma10 = mt.MA(close, 10)
            ma20 = mt.MA(close, 20)
            ma60 = mt.MA(close, 60)

            # ATR真实波动率 - 改进计算逻辑
            atr = mt.ATR(close, high, low, N=20)
            
            # EMV简易波动指标 - 增加除零保护
            emv, maemv = mt.EMV(high, low, volume, N=14, M=9)

            # 高级技术指标计算 - 使用优化算法
            dpo, madpo = mt.DPO(close, M1=20, M2=10, M3=6)  # 区间震荡线
            trix, trma = mt.TRIX(close, M1=12, M2=20)  # 三重指数平滑平均
            pdi, mdi, adx, adxr = mt.DMI(close, high, low, M1=14, M2=6)  # 动向指标
            vr = mt.VR(close, volume, M1=26)  # 成交量比率
            ar, br = mt.BRAR(open_price, close, high, low, M1=26)  # 人气意愿指标
            roc, maroc = mt.ROC(close, N=12, M=6)  # 变动率指标
            mtm, mtmma = mt.MTM(close, N=12, M=6)  # 动量指标
            dif_dma, difma_dma = mt.DMA(close, N1=10, N2=50, M=10)  # 平行线差指标
            
            # 新增高级指标
            obv = mt.OBV(close, volume)  # 能量潮指标
            bbi = mt.BBI(close, M1=3, M2=6, M3=12, M4=20)  # 多空指标
            
            print(f"技术指标计算完成，共计算了 20+ 个指标")

            # 保存基础技术指标到DataFrame
            df['MACD'] = macd
            df['DIF'] = dif
            df['DEA'] = dea
            df['K'] = k
            df['D'] = d
            df['J'] = j
            df['BOLL_UPPER'] = upper
            df['BOLL_MID'] = mid
            df['BOLL_LOWER'] = lower
            df['RSI'] = rsi
            df['PSY'] = psy
            df['PSYMA'] = psyma
            df['WR'] = wr
            df['WR1'] = wr1
            df['BIAS1'] = bias1
            df['BIAS2'] = bias2
            df['BIAS3'] = bias3
            df['CCI'] = cci
            
            # 保存移动平均线指标
            df['MA5'] = ma5
            df['MA10'] = ma10
            df['MA20'] = ma20
            df['MA60'] = ma60
            
            # 保存波动性和成交量指标
            df['ATR'] = atr
            df['EMV'] = emv
            df['MAEMV'] = maemv
            df['OBV'] = obv  # 新增能量潮指标
            df['BBI'] = bbi  # 新增多空指标
            
            # 保存高级技术指标
            df['DPO'] = dpo
            df['MADPO'] = madpo
            df['TRIX'] = trix
            df['TRMA'] = trma
            df['PDI'] = pdi
            df['MDI'] = mdi
            df['ADX'] = adx
            df['ADXR'] = adxr
            df['VR'] = vr
            df['AR'] = ar
            df['BR'] = br
            df['ROC'] = roc
            df['MAROC'] = maroc
            df['MTM'] = mtm
            df['MTMMA'] = mtmma
            df['DIF_DMA'] = dif_dma
            df['DIFMA_DMA'] = difma_dma

            return df

        except Exception as e:
            print(f"计算技术指标时出错: {str(e)}")
            return None
    
    def generate_trading_signals(self, code):
        """生成交易信号 - 调用优化后的全局函数"""
        if code not in self.data:
            return None
        
        return generate_trading_signals(self.data[code])

    def plot_analysis(self, code):
        """
        将原复合图表拆分为四个独立的图表，每个图表单独渲染
        美化版本：增强视觉效果、配色方案和交互体验
        """
        if code not in self.data:
            print(f"错误: 无法绘制图表，股票代码 {code} 没有数据")
            return None

        df = self.calculate_indicators(code)
        if df is None:
            print(f"错误: 无法计算技术指标，跳过图表生成")
            return None

        stock_name = self.get_stock_name(code)

        try:
            # 定义专业的配色方案
            colors = {
                'close': '#2E86AB',  # 深蓝色 - 收盘价
                'ma5': '#A23B72',  # 玫瑰红 - MA5
                'ma10': '#F18F01',  # 橙色 - MA10
                'ma20': '#C73E1D',  # 深红色 - MA20
                'boll': '#6C757D',  # 灰色 - 布林带
                'macd_pos': '#28A745',  # 绿色 - MACD正值
                'macd_neg': '#DC3545',  # 红色 - MACD负值
                'dif': '#FF6B35',  # 橙红色 - DIF
                'dea': '#7209B7',  # 紫色 - DEA
                'k': '#0D6EFD',  # 蓝色 - K线
                'd': '#FD7E14',  # 橙色 - D线
                'j': '#198754',  # 绿色 - J线
                'rsi': '#6F42C1',  # 深紫色 - RSI
                'overbought': '#DC3545',  # 红色 - 超买线
                'oversold': '#198754'  # 绿色 - 超卖线
            }

            charts_html = []

            # 1. 价格走势与技术指标图
            price_fig = go.Figure()

            # 收盘价
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['close'],
                name='收盘价',
                line=dict(color=colors['close'], width=3),
                hovertemplate='<b>收盘价</b><br>日期: %{x}<br>价格: ¥%{y:.2f}<extra></extra>'
            ))

            # 移动平均线
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MA5'],
                name='MA5',
                line=dict(color=colors['ma5'], width=2, dash='solid'),
                hovertemplate='<b>MA5</b><br>日期: %{x}<br>价格: ¥%{y:.2f}<extra></extra>'
            ))

            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MA10'],
                name='MA10',
                line=dict(color=colors['ma10'], width=2, dash='solid'),
                hovertemplate='<b>MA10</b><br>日期: %{x}<br>价格: ¥%{y:.2f}<extra></extra>'
            ))

            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MA20'],
                name='MA20',
                line=dict(color=colors['ma20'], width=2, dash='solid'),
                hovertemplate='<b>MA20</b><br>日期: %{x}<br>价格: ¥%{y:.2f}<extra></extra>'
            ))

            # 布林带
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BOLL_UPPER'],
                name='布林上轨',
                line=dict(color=colors['boll'], width=1, dash='dot'),
                hovertemplate='<b>布林上轨</b><br>日期: %{x}<br>价格: ¥%{y:.2f}<extra></extra>'
            ))

            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BOLL_LOWER'],
                name='布林下轨',
                line=dict(color=colors['boll'], width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(108, 117, 125, 0.1)',
                hovertemplate='<b>布林下轨</b><br>日期: %{x}<br>价格: ¥%{y:.2f}<extra></extra>'
            ))

            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BOLL_MID'],
                name='布林中轨',
                line=dict(color=colors['boll'], width=1, dash='dash'),
                hovertemplate='<b>布林中轨</b><br>日期: %{x}<br>价格: ¥%{y:.2f}<extra></extra>'
            ))

            # 添加当前价格注释
            current_price = df['close'].iloc[-1]
            price_fig.add_annotation(
                x=df.index[-1],
                y=current_price,
                text=f"当前价格<br>¥{current_price:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#2E86AB",
                bgcolor="rgba(46, 134, 171, 0.8)",
                bordercolor="#2E86AB",
                borderwidth=2,
                font=dict(color="white", size=10)
            )

            price_fig.update_layout(
                title=f'📈 {stock_name} ({code}) 价格走势与技术指标',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                ),
                hovermode='x unified',
                template='plotly_white',
                paper_bgcolor='#FAFAFA',
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='%Y-%m-%d'
                ),
                yaxis=dict(
                    title="价格 (¥)",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='.2f'
                )
            )

            charts_html.append(price_fig.to_html(
                include_plotlyjs='cdn',
                full_html=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{stock_name}_{code}_price_analysis',
                        'height': 500,
                        'width': 1000,
                        'scale': 2
                    }
                }
            ))

            # 2. MACD指标图
            macd_fig = go.Figure()

            # MACD柱状图
            macd_colors = [colors['macd_pos'] if x >= 0 else colors['macd_neg'] for x in df['MACD']]
            macd_fig.add_trace(go.Bar(
                x=df.index,
                y=df['MACD'],
                name='MACD柱',
                marker_color=macd_colors,
                marker_line=dict(width=0),
                opacity=0.8,
                hovertemplate='<b>MACD</b><br>日期: %{x}<br>值: %{y:.4f}<extra></extra>'
            ))

            macd_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['DIF'],
                name='DIF快线',
                line=dict(color=colors['dif'], width=2),
                hovertemplate='<b>DIF</b><br>日期: %{x}<br>值: %{y:.4f}<extra></extra>'
            ))

            macd_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['DEA'],
                name='DEA慢线',
                line=dict(color=colors['dea'], width=2),
                hovertemplate='<b>DEA</b><br>日期: %{x}<br>值: %{y:.4f}<extra></extra>'
            ))

            macd_fig.update_layout(
                title=f'📊 {stock_name} ({code}) MACD指标',
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                ),
                hovermode='x unified',
                template='plotly_white',
                paper_bgcolor='#FAFAFA',
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='%Y-%m-%d'
                ),
                yaxis=dict(
                    title="MACD",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='.4f'
                )
            )

            charts_html.append(macd_fig.to_html(
                include_plotlyjs=False,
                full_html=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{stock_name}_{code}_macd_analysis',
                        'height': 400,
                        'width': 1000,
                        'scale': 2
                    }
                }
            ))

            # 3. KDJ随机指标图
            kdj_fig = go.Figure()

            kdj_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['K'],
                name='K值',
                line=dict(color=colors['k'], width=2.5),
                hovertemplate='<b>K值</b><br>日期: %{x}<br>值: %{y:.2f}<extra></extra>'
            ))

            kdj_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['D'],
                name='D值',
                line=dict(color=colors['d'], width=2.5),
                hovertemplate='<b>D值</b><br>日期: %{x}<br>值: %{y:.2f}<extra></extra>'
            ))

            kdj_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['J'],
                name='J值',
                line=dict(color=colors['j'], width=2.5),
                hovertemplate='<b>J值</b><br>日期: %{x}<br>值: %{y:.2f}<extra></extra>'
            ))

            # 添加KDJ参考线
            kdj_fig.add_hline(y=80, line=dict(color='rgba(220, 53, 69, 0.5)', dash='dash', width=1))
            kdj_fig.add_hline(y=20, line=dict(color='rgba(25, 135, 84, 0.5)', dash='dash', width=1))

            kdj_fig.update_layout(
                title=f'📉 {stock_name} ({code}) KDJ随机指标',
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                ),
                hovermode='x unified',
                template='plotly_white',
                paper_bgcolor='#FAFAFA',
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='%Y-%m-%d'
                ),
                yaxis=dict(
                    title="KDJ (%)",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='.2f'
                )
            )

            charts_html.append(kdj_fig.to_html(
                include_plotlyjs=False,
                full_html=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{stock_name}_{code}_kdj_analysis',
                        'height': 400,
                        'width': 1000,
                        'scale': 2
                    }
                }
            ))

            # 4. RSI相对强弱指标图
            rsi_fig = go.Figure()

            rsi_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color=colors['rsi'], width=3),
                hovertemplate='<b>RSI</b><br>日期: %{x}<br>值: %{y:.2f}<extra></extra>'
            ))

            # RSI参考线和区域
            rsi_fig.add_hline(y=70, line=dict(color=colors['overbought'], dash='dash', width=2))
            rsi_fig.add_hline(y=30, line=dict(color=colors['oversold'], dash='dash', width=2))

            # 添加RSI超买超卖区域填充
            rsi_fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220, 53, 69, 0.1)", line_width=0)
            rsi_fig.add_hrect(y0=0, y1=30, fillcolor="rgba(25, 135, 84, 0.1)", line_width=0)

            rsi_fig.update_layout(
                title=f'📋 {stock_name} ({code}) RSI相对强弱指标',
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                ),
                hovermode='x unified',
                template='plotly_white',
                paper_bgcolor='#FAFAFA',
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='%Y-%m-%d'
                ),
                yaxis=dict(
                    title="RSI",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='.2f',
                    range=[0, 100]
                )
            )

            charts_html.append(rsi_fig.to_html(
                include_plotlyjs=False,
                full_html=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{stock_name}_{code}_rsi_analysis',
                        'height': 400,
                        'width': 1000,
                        'scale': 2
                    }
                }
            ))

            # 将所有图表HTML合并
            combined_html = '\n'.join(charts_html)
            return combined_html

        except Exception as e:
            print(f"生成交互式图表时出错: {str(e)}")
            return None

    def generate_analysis_data(self, code):
        """生成股票分析数据"""
        if code not in self.data:
            print(f"错误: 股票代码 {code} 没有数据，无法生成分析")
            stock_name = self.get_stock_name(code)
            return {
                "基础数据": {
                    "股票代码": code,
                    "股票名称": stock_name,
                    "数据状态": "数据获取失败",
                    "错误信息": "请检查股票代码格式是否正确"
                },
                "技术指标": {},
                "技术分析建议": ["数据获取失败，无法进行技术分析"]
            }

        df = self.data[code]

        # 检查数据是否为空
        if df.empty:
            print(f"错误: 股票代码 {code} 数据为空")
            stock_name = self.get_stock_name(code)
            return {
                "基础数据": {
                    "股票代码": code,
                    "股票名称": stock_name,
                    "数据状态": "数据为空",
                    "错误信息": "获取到的数据为空，请检查股票代码"
                },
                "技术指标": {},
                "技术分析建议": ["数据为空，无法进行分析"]
            }

        latest_df = self.calculate_indicators(code)

        if latest_df is None:
            print(f"错误: 股票代码 {code} 技术指标计算失败")
            stock_name = self.get_stock_name(code)
            return {
                "基础数据": {
                    "股票代码": code,
                    "股票名称": stock_name,
                    "数据状态": "技术指标计算失败"
                },
                "技术指标": {},
                "技术分析建议": ["技术指标计算失败"]
            }

        try:
            analysis_data = {
                "基础数据": {
                    "股票代码": code,
                    "最新收盘价": f"{df['close'].iloc[-1]:.2f}",
                    "涨跌幅": f"{((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100):.2f}%",
                    "最高价": f"{df['high'].iloc[-1]:.2f}",
                    "最低价": f"{df['low'].iloc[-1]:.2f}",
                    "成交量": f"{int(df['volume'].iloc[-1]):,}",
                },
                "技术指标": {
                    "MA指标": {
                        "MA5": f"{latest_df['MA5'].iloc[-1]:.2f}",
                        "MA10": f"{latest_df['MA10'].iloc[-1]:.2f}",
                        "MA20": f"{latest_df['MA20'].iloc[-1]:.2f}",
                        "MA60": f"{latest_df['MA60'].iloc[-1]:.2f}",
                    },
                    "趋势指标": {
                        "MACD (指数平滑异同移动平均线)": f"{latest_df['MACD'].iloc[-1]:.2f}",
                        "DIF (差离值)": f"{latest_df['DIF'].iloc[-1]:.2f}",
                        "DEA (讯号线)": f"{latest_df['DEA'].iloc[-1]:.2f}",
                        "TRIX (三重指数平滑平均线)": f"{latest_df['TRIX'].iloc[-1]:.2f}",
                        "PDI (上升方向线)": f"{latest_df['PDI'].iloc[-1]:.2f}",
                        "MDI (下降方向线)": f"{latest_df['MDI'].iloc[-1]:.2f}",
                        "ADX (趋向指标)": f"{latest_df['ADX'].iloc[-1]:.2f}",
                    },
                    "摆动指标": {
                        "RSI (相对强弱指标)": f"{latest_df['RSI'].iloc[-1]:.2f}",
                        "KDJ-K (随机指标K值)": f"{latest_df['K'].iloc[-1]:.2f}",
                        "KDJ-D (随机指标D值)": f"{latest_df['D'].iloc[-1]:.2f}",
                        "KDJ-J (随机指标J值)": f"{latest_df['J'].iloc[-1]:.2f}",
                        "BIAS (乖离率)": f"{latest_df['BIAS1'].iloc[-1]:.2f}",
                        "CCI (顺势指标)": f"{latest_df['CCI'].iloc[-1]:.2f}",
                    },
                    "成交量指标": {
                        "VR (成交量比率)": f"{latest_df['VR'].iloc[-1]:.2f}",
                        "AR (人气指标)": f"{latest_df['AR'].iloc[-1]:.2f}",
                        "BR (意愿指标)": f"{latest_df['BR'].iloc[-1]:.2f}",
                    },
                    "动量指标": {
                        "ROC (变动率)": f"{latest_df['ROC'].iloc[-1]:.2f}",
                        "MTM (动量指标)": f"{latest_df['MTM'].iloc[-1]:.2f}",
                        "DPO (区间振荡)": f"{latest_df['DPO'].iloc[-1]:.2f}",
                    },
                    "布林带": {
                        "BOLL上轨": f"{latest_df['BOLL_UPPER'].iloc[-1]:.2f}",
                        "BOLL中轨": f"{latest_df['BOLL_MID'].iloc[-1]:.2f}",
                        "BOLL下轨": f"{latest_df['BOLL_LOWER'].iloc[-1]:.2f}",
                    },
                    "RPS相对强度": {
                        "最新RPS值": f"{self.get_latest_rps(code) or 'N/A'}",
                        "RPS说明": "RPS值越高表示相对强度越强，80以上为强势股"
                    }
                },
                "技术分析建议": generate_trading_signals(latest_df)
            }

            """添加AI分析结果"""
            if self.llm:
                try:
                    print("正在调用AI进行智能分析...")
                    stock_name = self.get_stock_name(code)
                    api_result = self.llm.request_analysis(df, latest_df, stock_code=code, stock_name=stock_name)
                    if api_result:
                        # 保存原始输出到文件
                        if '_raw_analysis_text' in api_result:
                            raw_text = api_result['_raw_analysis_text']
                            self.save_raw_analysis_to_file(code, raw_text)
                            # 从结果中移除原始文本，避免在HTML中显示
                            del api_result['_raw_analysis_text']
                        
                        analysis_data.update(api_result)
                        print("AI分析完成")
                    else:
                        print("AI分析未返回结果")
                except Exception as e:
                    print(f"AI分析过程出错: {str(e)}")
            else:
                print("未配置LLM API，跳过AI分析")

            return analysis_data

        except Exception as e:
            print(f"生成分析数据时出错: {str(e)}")
            stock_name = self.get_stock_name(code)
            return {
                "基础数据": {
                    "股票代码": code,
                    "股票名称": stock_name,
                    "数据状态": f"分析出错: {str(e)}"
                },
                "技术指标": {},
                "技术分析建议": [f"分析出错: {str(e)}"]
            }

    def _generate_ai_analysis_html(self, ai_analysis):
        """生成AI分析结果的HTML代码"""
        html = """
        <div class="ai-analysis-section">
            <h3>AI智能分析结果</h3>
            <div class="analysis-grid">
        """

        # 添加各个分析部分
        for section_name, content in ai_analysis.items():
            if section_name == "分析状态" and content == "分析失败":
                continue
            html += f"""
                <div class="analysis-card">
                    <h4>{section_name}</h4>
                    {self._format_analysis_content(content)}
                </div>
            """

        html += """
            </div>
        </div>
        """
        return html

    def _format_analysis_content(self, content):
        """格式化分析内容为HTML"""
        if isinstance(content, dict):
            html = "<table class='analysis-table'>"
            for key, value in content.items():
                html += f"<tr><td>{key}</td><td>{self._format_analysis_content(value)}</td></tr>"
            html += "</table>"
            return html
        elif isinstance(content, list):
            return "<ul>" + "".join(f"<li>{item}</li>" for item in content) + "</ul>"
        else:
            return str(content)

    def generate_html_report(self):
        """生成HTML格式的分析报告"""
        # 检查模板文件是否存在
        template_path = 'static/templates/report_template.html'
        css_path = 'static/css/report.css'

        if not os.path.exists(template_path):
            print(f"错误: 模板文件不存在: {template_path}")
            print("请创建模板文件或使用简化版本")
            return self.generate_simple_html_report()

        if not os.path.exists(css_path):
            print(f"警告: CSS文件不存在: {css_path}")
            css_content = "/* 默认样式 */"
        else:
            # 读取样式文件
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()

        # 读取模板文件
        with open(template_path, 'r', encoding='utf-8') as f:
            html_template = f.read()

        tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(tz).strftime('%Y年%m月%d日 %H时%M分%S秒')

        stock_contents = []
        for code in self.stock_codes:
            if code in self.data:
                analysis_data = self.generate_analysis_data(code)
                chart_base64 = self.plot_analysis(code)
                stock_name = self.get_stock_name(code)

                # 生成基础数据部分的HTML
                basic_data_html = f"""
                <div class="indicator-section">
                    <h3>基础数据</h3>
                    <table class="data-table">
                        <tr>
                            <th>指标</th>
                            <th>数值</th>
                        </tr>
                        {''.join(_generate_table_row(k, v) for k, v in analysis_data['基础数据'].items())}
                    </table>
                </div>
                """

                # 生成技术指标部分的HTML
                indicator_sections = []
                for section_name, indicators in analysis_data['技术指标'].items():
                    indicator_html = f"""
                    <div class="indicator-section">
                        <h3>{section_name}</h3>
                        <table class="data-table">
                            <tr>
                                <th>指标</th>
                                <th>数值</th>
                            </tr>
                            {''.join(_generate_table_row(k, v) for k, v in indicators.items())}
                        </table>
                    </div>
                    """
                    indicator_sections.append(indicator_html)

                # 生成交易信号部分的HTML
                signals_html = f"""
                <div class="indicator-section">
                    <h3>交易信号</h3>
                    <ul class="signal-list">
                        {''.join(f'<li>{signal}</li>' for signal in analysis_data['技术分析建议'])}
                    </ul>
                </div>
                """

                # 生成AI分析结果的HTML
                ai_analysis_html = ""
                if "AI分析结果" in analysis_data:
                    sections = analysis_data["AI分析结果"]
                    for section_name, content in sections.items():
                        if section_name != "分析状态":
                            ai_analysis_html += f"""
                            <div class="indicator-section">
                                <h3>{section_name}</h3>
                                <div class="analysis-content">
                                    {content}
                                </div>
                            </div>
                            """

                # 图表部分
                chart_html = self.plot_analysis(code)

                if chart_base64:
                    chart_html = f"""
                    <div class="section-divider">
                        <h2>技术指标图表</h2>
                    </div>
                    <div class="chart-container">
                        {chart_html}
                    </div>
                    """
                else:
                    chart_html = f"""
                    <div class="section-divider">
                        <h2>技术指标图表</h2>
                    </div>
                    
                    <div class="chart-container">
                        <p>图表生成失败，请检查数据和配置</p>
                    </div>
                    """

                # 组合单个股票的完整内容
                stock_content = f"""
                <div class="stock-container">
                    <h2>{stock_name} ({code}) 分析报告</h2>
                    
                    <div class="section-divider">
                        <h2>基础技术分析</h2>
                    </div>
                    
                    <div class="data-grid">
                        {basic_data_html}
                        {signals_html}
                    </div>
                    
                    <div class="section-divider">
                        <h2>技术指标详情</h2>
                    </div>
                    
                    {''.join(indicator_sections)}
                    
                    {chart_html}
            
                    <div class="section-divider">
                        <h2>人工智能分析报告</h2>
                    </div>
                    {ai_analysis_html}
                </div>
                """
                stock_contents.append(stock_content)
            else:
                stock_name = self.get_stock_name(code)
                stock_content = f"""
                <div class="stock-container">
                    <h2>{stock_name} ({code}) 分析报告</h2>
                    <div class="error-message">
                        <h3>数据获取失败</h3>
                        <p>无法获取股票 {stock_name} ({code}) 的数据</p>
                        <p>请检查股票代码格式是否正确：</p>
                        <ul>
                            <li>上交所: sh000001 (上证指数), sh600036 (招商银行)</li>
                            <li>深交所: sz399001 (深证成指), sz000001 (平安银行), sz002640 (跨境通)</li>
                        </ul>
                    </div>
                </div>
                """
                stock_contents.append(stock_content)

        # 将CSS样式和内容插入到模板中
        template = Template(html_template)
        html_content = template.substitute(
            styles=css_content,
            generate_time=current_time,
            content='\n'.join(stock_contents)
        )
        return html_content

    def generate_simple_html_report(self):
        """生成简化版HTML报告（当模板文件不存在时使用）"""
        tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(tz).strftime('%Y年%m月%d日 %H时%M分%S秒')

        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>股票技术分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .stock-container {{ margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .data-table th {{ background-color: #f5f5f5; }}
                .positive {{ color: red; }}
                .negative {{ color: green; }}
                .neutral {{ color: black; }}
                .error-message {{ color: red; padding: 20px; background: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>股票技术分析报告</h1>
                <p>生成时间: {current_time}</p>
            </div>
        """

        for code in self.stock_codes:
            stock_name = self.get_stock_name(code)
            if code in self.data:
                analysis_data = self.generate_analysis_data(code)
                html_content += f"""
                <div class="stock-container">
                    <h2>{stock_name} ({code})</h2>
                    <h3>基础数据</h3>
                    <table class="data-table">
                        <tr><th>指标</th><th>数值</th></tr>
                        {''.join(_generate_table_row(k, v) for k, v in analysis_data['基础数据'].items())}
                    </table>
                    
                    <h3>技术分析建议</h3>
                    <ul>
                        {''.join(f'<li>{signal}</li>' for signal in analysis_data['技术分析建议'])}
                    </ul>
                </div>
                """
            else:
                html_content += f"""
                <div class="stock-container">
                    <h2>{stock_name} ({code})</h2>
                    <div class="error-message">
                        <p>数据获取失败，请检查股票代码是否正确。</p>
                        <p>常见股票代码格式：</p>
                        <ul>
                            <li>上交所: sh000001, sh600036</li>
                            <li>深交所: sz399001, sz000001, sz002640</li>
                        </ul>
                    </div>
                </div>
                """

        html_content += """
        </body>
        </html>
        """

        return html_content

    def run_analysis(self, output_path='public/index.html'):
        """运行分析并生成报告"""
        print("开始运行股票分析...")

        # 获取数据
        print("步骤1: 获取股票数据")
        self.fetch_data()

        # 检查是否有有效数据
        if not self.data:
            print("错误: 没有获取到任何有效的股票数据")
            print("请检查股票代码格式，常见格式：")
            print("  上交所: sh000001 (上证指数), sh600036 (招商银行)")
            print("  深交所: sz399001 (深证成指), sz000001 (平安银行), sz002640 (跨境通)")
            return None

        print(f"成功获取 {len(self.data)} 只股票的数据")

        # 生成报告
        print("步骤2: 生成HTML报告")
        try:
            html_report = self.generate_html_report()
        except Exception as e:
            print(f"生成HTML报告时出错: {str(e)}")
            return None

        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 写入HTML报告
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            print(f"分析报告已生成: {output_path}")
            return output_path
        except Exception as e:
            print(f"写入报告文件时出错: {str(e)}")
            return None


if __name__ == "__main__":
    # 从环境变量获取股票配置
    stocks_config_str = os.environ.get('STOCKS_CONFIG')
    stocks_config_self_str = os.environ.get('STOCKS_CONFIG_SELF')
    count = int(os.environ.get('COUNT', 120))
    
    stock_info = {}
    
    # 处理STOCKS_CONFIG
    if stocks_config_str:
        try:
            # 解析JSON格式的股票配置
            config_stocks = json.loads(stocks_config_str)
            stock_info.update(config_stocks)
            print(f"从 STOCKS_CONFIG 加载股票配置: {config_stocks}")
        except json.JSONDecodeError as e:
            print(f"解析 STOCKS_CONFIG 失败: {e}")
    
    # 处理STOCKS_CONFIG_SELF
    if stocks_config_self_str:
        try:
            # 解析JSON格式的自选股票配置
            config_self_stocks = json.loads(stocks_config_self_str)
            stock_info.update(config_self_stocks)
            print(f"从 STOCKS_CONFIG_SELF 加载股票配置: {config_self_stocks}")
        except json.JSONDecodeError as e:
            print(f"解析 STOCKS_CONFIG_SELF 失败: {e}")
    
    # 如果没有任何配置，使用默认配置
    if not stock_info:
        print("未找到任何股票配置环境变量，使用默认股票配置")
        stock_info = {
            '上证指数': 'sh000001'
        }

    print("开始股票技术分析...")
    print(f"分析股票: {list(stock_info.keys())}")
    print(f"数据获取条数: {count}")

    # 从环境变量获取LLM配置
    llm_api_key = os.environ.get('LLM_API_KEY')
    llm_base_url = os.environ.get('LLM_BASE_URL')
    llm_model = os.environ.get('LLM_MODEL')
    
    if llm_api_key and llm_base_url:
        print(f"LLM配置: {llm_base_url} - {llm_model}")
    else:
        print("警告: 未配置LLM API，将跳过AI分析")

    analyzer = StockAnalyzer(
        stock_info, 
        count=count,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model
    )
    report_path = analyzer.run_analysis()

    if report_path:
        print(f"✅ 分析完成！报告已保存到: {report_path}")
    else:
        print("❌ 分析失败，请检查错误信息")