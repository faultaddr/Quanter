"""
XGBoost模型训练器

用于股票涨跌预测的XGBoost模型训练和预测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 尝试导入XGBoost
try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectFromModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelPerformance:
    """模型性能指标"""
    accuracy: float
    precision: float
    recall: float
    auc: float
    cv_scores: Dict[str, List[float]]


class XGBoostTrainer:
    """
    XGBoost模型训练器

    支持时间序列交叉验证和特征选择
    """

    def __init__(
        self,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        use_feature_selection: bool = True,
        feature_selection_threshold: int = 50,
        random_state: Optional[int] = None  # 默认不固定随机种子
    ):
        """
        初始化XGBoost训练器

        Args:
            max_depth: 树的最大深度
            learning_rate: 学习率
            n_estimators: 树的数量
            subsample: 样本采样比例
            colsample_bytree: 特征采样比例
            min_child_weight: 最小子节点权重
            gamma: 最小分裂增益
            reg_alpha: L1正则化
            reg_lambda: L2正则化
            use_feature_selection: 是否使用特征选择
            feature_selection_threshold: 特征选择数量阈值
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost未安装，请运行: pip install xgboost scikit-learn")

        self.params = {
            'objective': 'binary:logistic',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'eval_metric': ['logloss', 'auc'],
            'tree_method': 'hist',
            'n_jobs': -1,
        }
        # 关键修复：XGBoost默认random_state=0会导致每次结果一致
        # 解决方案：使用时间戳生成随机种子，确保每次训练有随机性
        import time
        if random_state is not None:
            self.params['random_state'] = random_state
        else:
            # 使用时间戳+进程ID生成随机种子，确保每次不同
            self.params['random_state'] = int(time.time() * 1000) % (2**31)

        self.use_feature_selection = use_feature_selection
        self.feature_selection_threshold = feature_selection_threshold

        self.model = None
        self.scaler = StandardScaler()
        self.selected_features: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.performance: Optional[ModelPerformance] = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        early_stopping_rounds: int = 50
    ) -> 'XGBoostTrainer':
        """
        训练模型

        Args:
            X: 特征DataFrame
            y: 标签Series
            n_splits: 交叉验证折数
            early_stopping_rounds: 早停轮数

        Returns:
            self
        """
        # 重置状态，确保不使用缓存
        self.model = None
        self.selected_features = []
        self.feature_importance = None
        self.performance = None

        logger.info(f"开始训练模型，样本数: {len(X)}, 特征数: {len(X.columns)}")
        logger.info(f"模型参数: n_estimators={self.params['n_estimators']}, "
                   f"max_depth={self.params['max_depth']}, "
                   f"learning_rate={self.params['learning_rate']}, "
                   f"random_state={self.params.get('random_state', 'None')}")

        # 标准化特征
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # 特征选择
        if self.use_feature_selection:
            X_scaled = self._select_features(X_scaled, y)
            logger.info(f"特征选择后保留 {len(self.selected_features)} 个特征")

        # 时间序列交叉验证
        cv_results = self._cross_validate(X_scaled, y, n_splits, early_stopping_rounds)

        # 在全量数据上训练最终模型
        self._train_final_model(X_scaled, y, early_stopping_rounds)

        # 保存性能指标
        self.performance = ModelPerformance(
            accuracy=np.mean(cv_results['accuracy']),
            precision=np.mean(cv_results['precision']),
            recall=np.mean(cv_results['recall']),
            auc=np.mean(cv_results['auc']),
            cv_scores=cv_results
        )

        logger.info(f"模型训练完成，AUC: {self.performance.auc:.4f}")
        return self

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """特征选择"""
        temp_model = xgb.XGBClassifier(**{k: v for k, v in self.params.items() if k != 'n_estimators'}, n_estimators=100)
        temp_model.fit(X, y)

        # 获取特征重要性
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 选择top特征
        self.selected_features = importance_df.head(self.feature_selection_threshold)['feature'].tolist()
        self.feature_importance = importance_df

        return X[self.selected_features]

    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int,
        early_stopping_rounds: int
    ) -> Dict[str, List[float]]:
        """时间序列交叉验证"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'auc': []
        }

        logger.info("进行时间序列交叉验证...")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 处理类别不平衡
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

            model = xgb.XGBClassifier(
                **self.params,
                scale_pos_weight=scale_pos_weight,
                early_stopping_rounds=early_stopping_rounds
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            cv_results['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            cv_results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            cv_results['auc'].append(roc_auc_score(y_test, y_prob))

            logger.info(f"  Fold {fold + 1}: AUC={cv_results['auc'][-1]:.4f}, Acc={cv_results['accuracy'][-1]:.4f}")

        return cv_results

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, early_stopping_rounds: int):
        """训练最终模型"""
        scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1

        self.model = xgb.XGBClassifier(
            **self.params,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=early_stopping_rounds
        )

        # 使用最后20%数据作为验证集
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征DataFrame

        Returns:
            预测标签
        """
        if self.model is None:
            raise ValueError("模型未训练")

        X_scaled = self._preprocess_features(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征DataFrame

        Returns:
            上涨概率
        """
        if self.model is None:
            raise ValueError("模型未训练")

        X_scaled = self._preprocess_features(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """特征预处理"""
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        if self.selected_features:
            missing_features = [f for f in self.selected_features if f not in X_scaled.columns]
            if missing_features:
                # 为缺失特征填充0
                for f in missing_features:
                    X_scaled[f] = 0
            X_scaled = X_scaled[self.selected_features]

        return X_scaled

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性

        Args:
            top_n: 返回前N个重要特征

        Returns:
            特征重要性DataFrame
        """
        if self.feature_importance is not None:
            return self.feature_importance.head(top_n)
        return pd.DataFrame()

    def get_performance(self) -> ModelPerformance:
        """获取模型性能"""
        return self.performance

    def save_model(self, filepath: str):
        """保存模型"""
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'params': self.params
        }
        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        import joblib
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.selected_features = model_data['selected_features']
        self.feature_importance = model_data['feature_importance']
        self.params = model_data['params']
        logger.info(f"模型已从 {filepath} 加载")