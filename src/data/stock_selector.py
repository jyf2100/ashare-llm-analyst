"""
股票选择模块

使用训练好的ML模型对股票进行预测并筛选符合条件的股票
"""

import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.base import AnalyzerBase
from src.core.config import Config, get_config
from src.core.logger import get_logger

logger = get_logger(__name__)

# 尝试导入机器学习库
try:
    import joblib
    import yaml
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("机器学习库不可用，选股功能将受限")


class StockSelector(AnalyzerBase):
    """
    ML股票选择器

    使用训练好的机器学习模型对股票进行预测，并筛选出符合条件的股票

    Example:
        >>> selector = StockSelector()
        >>> result = selector.select_stocks(top_n=10, min_return=5.0)
        >>> print(f"选中 {result['selected_count']} 只股票")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化股票选择器

        Args:
            config: 配置实例
        """
        super().__init__(config)
        self.models_dir = self.config.data.models_dir
        self.data_dir = self.config.data.data_dir
        self.rps_dir = self.config.data.rps_dir

        # 模型存储
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_scalers: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}

        # 结果存储
        self.ml_predictions: Dict[str, Dict] = {}

        # 加载训练好的模型
        self.load_trained_models()

    def load_trained_models(self) -> None:
        """
        加载训练好的模型和配置

        从 models/ 目录加载:
        - .pkl 文件: 模型和标准化器
        - .yaml 文件: 模型配置
        """
        if not ML_AVAILABLE:
            logger.warning("机器学习库不可用，跳过模型加载")
            return

        if not os.path.exists(self.models_dir):
            logger.warning(f"模型目录不存在: {self.models_dir}")
            return

        try:
            # 查找模型文件
            model_files = glob.glob(os.path.join(self.models_dir, "*.pkl"))
            config_files = glob.glob(os.path.join(self.models_dir, "*.yaml"))

            # 加载配置文件
            for config_file in config_files:
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)

                    config_name = os.path.basename(config_file).replace('.yaml', '')
                    self.model_configs[config_name] = config
                    logger.info(f"加载配置: {config_name}")
                except Exception as e:
                    logger.error(f"加载配置文件失败 {config_file}: {e}")

            # 加载模型文件
            for model_file in model_files:
                try:
                    model_name = os.path.basename(model_file).replace('.pkl', '')

                    # 区分模型和标准化器
                    if 'scaler_' in model_name:
                        # 这是标准化器
                        scaler = joblib.load(model_file)
                        scaler_key = model_name.replace('scaler_', '')
                        self.loaded_scalers[scaler_key] = scaler
                        logger.debug(f"加载标准化器: {scaler_key}")
                    else:
                        # 这是模型
                        model = joblib.load(model_file)
                        self.loaded_models[model_name] = model
                        logger.debug(f"加载模型: {model_name}")

                except Exception as e:
                    logger.error(f"加载模型文件失败 {model_file}: {e}")

            logger.info(f"模型加载完成: {len(self.loaded_models)} 个模型, "
                       f"{len(self.loaded_scalers)} 个标准化器, "
                       f"{len(self.model_configs)} 个配置")

        except Exception as e:
            logger.error(f"加载训练好的模型时出错: {e}")

    def _select_best_model(
        self,
        prediction_days: int = 5,
        task_type: str = "regression"
    ) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
        """
        选择最佳模型

        Args:
            prediction_days: 预测天数
            task_type: 任务类型 ("regression" 或 "classification")

        Returns:
            (模型, 标准化器, 模型名称) 元组
        """
        if not self.loaded_models:
            logger.warning("没有可用的训练好的模型")
            return None, None, None

        # 根据预测天数和任务类型筛选合适的模型
        suitable_models = []

        for model_name in self.loaded_models.keys():
            # 解析模型名称中的信息
            if f"{prediction_days}d" in model_name:
                # 检查任务类型匹配
                if task_type == "regression" and ("return" in model_name and "gt_" not in model_name):
                    suitable_models.append(model_name)
                elif task_type == "classification" and ("gt_" in model_name or "multi_class" in model_name):
                    suitable_models.append(model_name)

        if not suitable_models:
            # 如果没有完全匹配的模型，选择最接近的
            logger.warning(f"没有找到完全匹配 {prediction_days}d {task_type} 的模型")
            # 选择第一个可用模型作为回退
            if self.loaded_models:
                suitable_models = [list(self.loaded_models.keys())[0]]

        if not suitable_models:
            return None, None, None

        # 选择第一个匹配的模型
        best_model_name = suitable_models[0]
        model = self.loaded_models[best_model_name]
        scaler = self.loaded_scalers.get(best_model_name, None)

        logger.info(f"选择模型: {best_model_name}")
        return model, scaler, best_model_name

    def _load_stock_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载股票数据

        Returns:
            股票代码到DataFrame的映射
        """
        stock_data = {}

        if not os.path.exists(self.data_dir):
            logger.warning(f"股票数据目录不存在: {self.data_dir}")
            return stock_data

        # 查找所有CSV文件
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        for csv_file in csv_files:
            try:
                stock_code = os.path.basename(csv_file).replace('.csv', '')
                df = pd.read_csv(csv_file)

                if len(df) > 0:
                    stock_data[stock_code] = df
            except Exception as e:
                logger.error(f"加载股票数据失败 {csv_file}: {e}")

        logger.info(f"加载 {len(stock_data)} 只股票数据")
        return stock_data

    def _load_rps_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载RPS数据

        Returns:
            RPS类型到DataFrame的映射
        """
        rps_data = {}

        if not os.path.exists(self.rps_dir):
            logger.warning(f"RPS数据目录不存在: {self.rps_dir}")
            return rps_data

        # 查找所有CSV文件
        csv_files = glob.glob(os.path.join(self.rps_dir, "*.csv"))

        for csv_file in csv_files:
            try:
                rps_type = os.path.basename(csv_file).replace('.csv', '').replace('rps_', '')
                df = pd.read_csv(csv_file, index_col=0)

                if len(df) > 0:
                    rps_data[rps_type] = df
            except Exception as e:
                logger.error(f"加载RPS数据失败 {csv_file}: {e}")

        logger.info(f"加载 {len(rps_data)} 个RPS数据")
        return rps_data

    def _prepare_features(
        self,
        stock_code: str,
        stock_df: pd.DataFrame,
        rps_dfs: Dict[str, pd.DataFrame]
    ) -> Optional[np.ndarray]:
        """
        准备机器学习特征

        Args:
            stock_code: 股票代码
            stock_df: 股票数据
            rps_dfs: RPS数据字典

        Returns:
            特征数组，失败返回None
        """
        try:
            # 获取最新行数据
            latest = stock_df.iloc[-1]

            # 提取技术指标特征
            features = []

            # 价格相关特征
            features.append(latest.get('close', 0))
            features.append(latest.get('volume', 0))
            features.append(latest.get('amount', 0))

            # 技术指标特征
            if 'MACD' in stock_df.columns:
                features.append(latest.get('MACD', 0))
            if 'DIF' in stock_df.columns:
                features.append(latest.get('DIF', 0))
            if 'DEA' in stock_df.columns:
                features.append(latest.get('DEA', 0))
            if 'K' in stock_df.columns:
                features.append(latest.get('K', 50))
            if 'D' in stock_df.columns:
                features.append(latest.get('D', 50))
            if 'J' in stock_df.columns:
                features.append(latest.get('J', 50))
            if 'RSI' in stock_df.columns:
                features.append(latest.get('RSI', 50))

            # RPS特征
            for rps_type, rps_df in rps_dfs.items():
                if stock_code in rps_df.columns:
                    rps_value = rps_df[stock_code].iloc[-1]
                    features.append(rps_value)

            # 确保特征数量一致
            if len(features) == 0:
                return None

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"准备特征失败 {stock_code}: {e}")
            return None

    def predict_stocks(
        self,
        stock_data: Dict[str, pd.DataFrame],
        rps_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        对所有股票进行ML预测

        Args:
            stock_data: 股票数据字典
            rps_data: RPS数据字典

        Returns:
            预测结果字典 {stock_code: {predicted_return: float, ...}}
        """
        if not ML_AVAILABLE or not self.loaded_models:
            logger.warning("ML不可用或无模型，跳过预测")
            return {}

        predictions = {}

        # 选择最佳模型（5天回归预测）
        model, scaler, model_name = self._select_best_model(
            prediction_days=5,
            task_type="regression"
        )

        if model is None:
            logger.error("没有可用的模型")
            return {}

        for stock_code, stock_df in stock_data.items():
            try:
                # 准备特征
                features = self._prepare_features(stock_code, stock_df, rps_data)

                if features is None:
                    continue

                # 标准化特征
                if scaler is not None:
                    features = scaler.transform(features)

                # 预测
                predicted_return = model.predict(features)[0]

                predictions[stock_code] = {
                    'predicted_return_5d': float(predicted_return),
                    'model_name': model_name,
                }

            except Exception as e:
                logger.debug(f"预测股票失败 {stock_code}: {e}")
                continue

        logger.info(f"完成 {len(predictions)} 只股票的预测")
        return predictions

    def filter_stocks(
        self,
        predictions: Dict[str, Dict],
        top_n: int,
        min_return: float
    ) -> List[Tuple[str, float]]:
        """
        筛选符合条件的股票

        Args:
            predictions: 预测结果
            top_n: 返回前N只股票
            min_return: 最小收益率阈值（百分比）

        Returns:
            (股票代码, 预测收益率) 列表，按收益率降序排序
        """
        # 过滤预测收益率 > min_return 的股票
        qualified = []

        for stock_code, pred_data in predictions.items():
            predicted_return = pred_data.get('predicted_return_5d', 0)

            # 将预测的收益率转换为百分比
            return_pct = predicted_return * 100

            if return_pct > min_return:
                qualified.append((stock_code, return_pct))

        # 按收益率降序排序
        qualified.sort(key=lambda x: x[1], reverse=True)

        # 取前top_n只
        result = qualified[:top_n]

        logger.info(f"筛选出 {len(result)} 只股票 (收益率 > {min_return}%)")
        return result

    def _save_results(
        self,
        qualified_stocks: List[Tuple[str, float]],
        predictions: Dict[str, Dict]
    ) -> None:
        """
        保存筛选结果

        Args:
            qualified_stocks: 筛选后的股票列表
            predictions: 完整的预测结果
        """
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'selected_stocks': [
                    {'code': code, 'predicted_return': return_val}
                    for code, return_val in qualified_stocks
                ],
                'all_predictions': predictions,
            }

            output_file = os.path.join(
                self.config.work_dir,
                self.config.data.selection_results_file
            )

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"保存筛选结果到: {output_file}")

        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def select_stocks(
        self,
        top_n: int = 10,
        min_return: float = 5.0
    ) -> Dict[str, Any]:
        """
        执行选股流程

        Args:
            top_n: 返回前N只股票
            min_return: 最小收益率阈值（百分比）

        Returns:
            执行结果字典
        """
        logger.info(f"开始选股: top_n={top_n}, min_return={min_return}%")

        try:
            # 加载数据
            stock_data = self._load_stock_data()
            rps_data = self._load_rps_data()

            if not stock_data:
                logger.warning("没有股票数据，跳过选股")
                return {
                    "success": True,
                    "selected_count": 0,
                    "stocks": [],
                    "message": "没有股票数据"
                }

            # 预测
            predictions = self.predict_stocks(stock_data, rps_data)

            if not predictions:
                logger.warning("没有预测结果，跳过选股")
                return {
                    "success": True,
                    "selected_count": 0,
                    "stocks": [],
                    "message": "没有预测结果"
                }

            # 筛选
            qualified = self.filter_stocks(predictions, top_n, min_return)

            # 保存结果
            self._save_results(qualified, predictions)

            logger.info(f"选股完成: 选中 {len(qualified)} 只股票")

            return {
                "success": True,
                "selected_count": len(qualified),
                "stocks": qualified,
                "message": f"选中 {len(qualified)} 只股票"
            }

        except Exception as e:
            logger.error(f"选股失败: {e}")
            return {
                "success": False,
                "selected_count": 0,
                "stocks": [],
                "error": str(e)
            }
