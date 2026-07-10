"""
模型训练器单元测试

测试 ModelTrainerBase, RandomForestTrainer
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, '..')

import numpy as np
import joblib

from src.data.model_trainer import (
    ModelTrainerBase,
    RandomForestTrainer,
)


class TestModelTrainerBase:
    """ModelTrainerBase单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_model_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)

    def test_cannot_instantiate_abstract(self):
        """测试不能实例化抽象基类"""
        try:
            ModelTrainerBase(model_save_dir=self.temp_model_dir)
            assert False, "应该抛出异常"
        except TypeError:
            pass  # 预期的异常

    def test_concrete_implementation(self):
        """测试具体实现可以实例化"""

        class ConcreteTrainer(ModelTrainerBase):
            def train(self, X, y, **kwargs):
                return {"success": True}

            def predict(self, X):
                return np.array([1, 2, 3])

        trainer = ConcreteTrainer(model_save_dir=self.temp_model_dir)
        assert trainer is not None


class TestRandomForestTrainer:
    """RandomForestTrainer单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_model_dir = tempfile.mkdtemp()
        self.temp_training_dir = tempfile.mkdtemp()
        self.trainer = RandomForestTrainer(
            training_data_dir=self.temp_training_dir,
            model_save_dir=self.temp_model_dir,
        )

        # 创建模拟训练数据
        self._create_training_data()

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)
        if os.path.exists(self.temp_training_dir):
            shutil.rmtree(self.temp_training_dir)

    def _create_training_data(self):
        """创建模拟训练数据"""
        timestamp = "20241224"

        # 特征数据 (1000样本, 20特征)
        X = np.random.randn(1000, 20)
        np.save(os.path.join(self.temp_training_dir, f"features_{timestamp}.npy"), X)

        # 标签数据 - 二分类
        y_binary = np.random.randint(0, 2, 1000)
        np.save(
            os.path.join(self.temp_training_dir, f"labels_return_5d_gt_5pct_{timestamp}.npy"),
            y_binary
        )

        # 标签数据 - 连续值
        y_continuous = np.random.randn(1000) * 0.1
        np.save(
            os.path.join(self.temp_training_dir, f"labels_return_5d_continuous_{timestamp}.npy"),
            y_continuous
        )

    def test_initialization(self):
        """测试初始化"""
        assert self.trainer.model is None
        assert self.trainer.scaler is None
        assert self.trainer.model_params is not None
        assert self.trainer.model_params["n_estimators"] == 200

    def test_load_training_data(self):
        """测试加载训练数据"""
        X, y, label = self.trainer.load_training_data("return_5d_gt_5pct")

        assert X is not None
        assert y is not None
        assert label == "return_5d_gt_5pct"
        assert X.shape == (1000, 20)
        assert y.shape == (1000,)

    def test_load_training_data_nonexistent_label(self):
        """测试加载不存在的标签"""
        X, y, label = self.trainer.load_training_data("nonexistent_label")

        # X可能存在，但y和label应该是None
        assert y is None
        assert label is None

    def test_preprocess_data(self):
        """测试数据预处理"""
        # 创建包含NaN和Inf的数据
        X = np.array([
            [1, 2, 3],
            [np.nan, 5, 6],
            [7, np.inf, 9],
            [10, 11, -np.inf],
        ])
        y = np.array([0, 1, 0, 1])

        X_processed, y_processed = self.trainer.preprocess_data(X, y)

        # NaN应该被填充
        assert not np.isnan(X_processed).any()

        # Inf应该被处理
        assert not np.isinf(X_processed).any()

        # scaler应该被创建
        assert self.trainer.scaler is not None

    def test_preprocess_data_without_y(self):
        """测试不传入y的数据预处理"""
        X = np.random.randn(100, 10)

        X_processed, y_processed = self.trainer.preprocess_data(X, y=None)

        assert X_processed is not None
        assert y_processed is None

    def test_train_classifier(self):
        """测试训练分类器"""
        X = np.random.randn(500, 10)
        y = np.random.randint(0, 2, 500)

        results = self.trainer.train(X, y, model_type="classifier")

        assert results["model_type"] == "classifier"
        assert results["train_samples"] > 0
        assert results["test_samples"] > 0
        assert "accuracy" in results
        assert "f1" in results
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["f1"] <= 1

        # 模型应该被创建
        assert self.trainer.model is not None

    def test_train_regressor(self):
        """测试训练回归器"""
        X = np.random.randn(500, 10)
        y = np.random.randn(500) * 0.1

        results = self.trainer.train(X, y, model_type="regressor")

        assert results["model_type"] == "regressor"
        assert "rmse" in results
        assert "r2" in results
        assert "mae" in results
        assert results["rmse"] >= 0

        # 模型应该被创建
        assert self.trainer.model is not None

    def test_predict(self):
        """测试预测"""
        # 先训练模型
        X_train = np.random.randn(500, 10)
        y_train = np.random.randint(0, 2, 500)
        self.trainer.train(X_train, y_train, model_type="classifier")

        # 进行预测
        X_test = np.random.randn(10, 10)
        predictions, probabilities = self.trainer.predict(X_test)

        assert predictions is not None
        assert predictions.shape == (10,)
        assert probabilities is not None
        assert probabilities.shape == (10, 2)  # 二分类

    def test_predict_without_model(self):
        """测试未训练模型时预测"""
        trainer = RandomForestTrainer(model_save_dir=self.temp_model_dir)

        X = np.random.randn(10, 10)
        predictions, probabilities = trainer.predict(X)

        assert predictions is None
        assert probabilities is None

    def test_save_model(self):
        """测试保存模型"""
        # 创建并训练模型
        X = np.random.randn(500, 10)
        y = np.random.randint(0, 2, 500)
        self.trainer.train(X, y, model_type="classifier")

        # 保存模型
        success = self.trainer.save_model("test_model")

        assert success is True

        # 检查文件存在
        model_files = [f for f in os.listdir(self.temp_model_dir) if f.startswith("test_model_")]
        assert len(model_files) >= 1

        # 检查scaler也被保存
        scaler_files = [f for f in os.listdir(self.temp_model_dir) if f.startswith("scaler_test_model_")]
        assert len(scaler_files) == 1

    def test_load_model(self):
        """测试加载模型"""
        # 创建并训练模型
        X = np.random.randn(500, 10)
        y = np.random.randint(0, 2, 500)
        self.trainer.train(X, y, model_type="classifier")

        # 保存模型
        self.trainer.save_model("test_model")

        # 查找保存的模型文件
        model_files = [f for f in os.listdir(self.temp_model_dir) if f.startswith("test_model_") and not f.startswith("scaler_")]
        model_path = os.path.join(self.temp_model_dir, model_files[0])

        scaler_files = [f for f in os.listdir(self.temp_model_dir) if f.startswith("scaler_test_model_")]
        scaler_path = os.path.join(self.temp_model_dir, scaler_files[0])

        # 创建新的trainer并加载模型
        new_trainer = RandomForestTrainer(model_save_dir=self.temp_model_dir)
        success = new_trainer.load_model(model_path, scaler_path)

        assert success is True
        assert new_trainer.model is not None
        assert new_trainer.scaler is not None

    def test_train_models_convenience_method(self):
        """测试便捷的训练方法"""
        results = self.trainer.train_models("return_5d_gt_5pct")

        assert results is not None
        assert len(results) > 0

        for model_name, model_results in results.items():
            assert "random_forest" in model_name
            assert model_results["accuracy"] is not None


class TestModelEvaluation:
    """模型评估测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_model_dir = tempfile.mkdtemp()
        self.temp_training_dir = tempfile.mkdtemp()
        self.trainer = RandomForestTrainer(
            training_data_dir=self.temp_training_dir,
            model_save_dir=self.temp_model_dir,
        )

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)
        if os.path.exists(self.temp_training_dir):
            shutil.rmtree(self.temp_training_dir)

    def test_feature_importances(self):
        """测试特征重要性"""
        X = np.random.randn(500, 10)
        y = np.random.randint(0, 2, 500)

        results = self.trainer.train(X, y, model_type="classifier")

        assert "feature_importances" in results
        assert len(results["feature_importances"]) == 10

        # 特征重要性应该都是非负数
        assert all(imp >= 0 for imp in results["feature_importances"])

    def test_model_persistence(self):
        """测试模型持久化"""
        # 训练模型
        X_train = np.random.randn(500, 10)
        y_train = np.random.randint(0, 2, 500)
        self.trainer.train(X_train, y_train, model_type="classifier")

        # 保存前的预测
        X_test = np.random.randn(10, 10)
        pred1, _ = self.trainer.predict(X_test)

        # 保存并加载模型
        self.trainer.save_model("test_model")

        model_files = [f for f in os.listdir(self.temp_model_dir) if f.startswith("test_model_") and not f.startswith("scaler_")]
        scaler_files = [f for f in os.listdir(self.temp_model_dir) if f.startswith("scaler_test_model_")]

        new_trainer = RandomForestTrainer(model_save_dir=self.temp_model_dir)
        new_trainer.load_model(
            os.path.join(self.temp_model_dir, model_files[0]),
            os.path.join(self.temp_model_dir, scaler_files[0])
        )

        # 加载后的预测
        pred2, _ = new_trainer.predict(X_test)

        # 预测结果应该一致
        assert (pred1 == pred2).all()

    def test_classification_metrics(self):
        """测试分类指标"""
        X = np.random.randn(500, 10)
        y = np.random.randint(0, 2, 500)

        results = self.trainer.train(X, y, model_type="classifier")

        # 检查指标范围
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["f1"] <= 1
        if results["auc"] is not None:
            assert 0 <= results["auc"] <= 1

    def test_regression_metrics(self):
        """测试回归指标"""
        X = np.random.randn(500, 10)
        y = np.random.randn(500)

        results = self.trainer.train(X, y, model_type="regressor")

        # 检查指标合理性
        assert results["rmse"] >= 0
        assert -1 <= results["r2"] <= 1  # R²可能为负
        assert results["mae"] >= 0


if __name__ == "__main__":
    import pytest

    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
