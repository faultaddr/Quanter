"""
自选股管理器 - 管理用户的自选股列表
"""
import json
import os
import pandas as pd
from typing import List, Dict, Optional


class WatchlistManager:
    """
    自选股管理器，用于存储和管理用户关注的股票列表
    """

    def __init__(self, watchlist_file: str = "user_watchlist.json"):
        """
        初始化自选股管理器

        Args:
            watchlist_file: 自选股存储文件路径
        """
        self.watchlist_file = watchlist_file
        self.watchlists = self.load_watchlists()

    def load_watchlists(self) -> Dict[str, List[str]]:
        """
        从文件加载自选股列表

        Returns:
            包含多个自选股列表的字典
        """
        if os.path.exists(self.watchlist_file):
            try:
                with open(self.watchlist_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                # 文件损坏或不存在时返回空字典
                return {"default": []}
        else:
            # 默认创建一个空的自选股列表
            return {"default": []}

    def save_watchlists(self):
        """
        将自选股列表保存到文件
        """
        with open(self.watchlist_file, 'w', encoding='utf-8') as f:
            json.dump(self.watchlists, f, ensure_ascii=False, indent=2)

    def get_watchlist_names(self) -> List[str]:
        """
        获取所有自选股列表的名称

        Returns:
            自选股列表名称列表
        """
        return list(self.watchlists.keys())

    def get_watchlist(self, name: str = "default") -> List[str]:
        """
        获取指定名称的自选股列表

        Args:
            name: 自选股列表名称

        Returns:
            股票代码列表
        """
        return self.watchlists.get(name, [])

    def add_stock_to_watchlist(self, stock_code: str, watchlist_name: str = "default"):
        """
        向自选股列表添加股票

        Args:
            stock_code: 股票代码
            watchlist_name: 自选股列表名称
        """
        if watchlist_name not in self.watchlists:
            self.watchlists[watchlist_name] = []

        if stock_code not in self.watchlists[watchlist_name]:
            self.watchlists[watchlist_name].append(stock_code)
            self.save_watchlists()

    def remove_stock_from_watchlist(self, stock_code: str, watchlist_name: str = "default"):
        """
        从自选股列表移除股票

        Args:
            stock_code: 股票代码
            watchlist_name: 自选股列表名称
        """
        if watchlist_name in self.watchlists and stock_code in self.watchlists[watchlist_name]:
            self.watchlists[watchlist_name].remove(stock_code)
            self.save_watchlists()

    def create_watchlist(self, name: str, stocks: List[str] = None):
        """
        创建新的自选股列表

        Args:
            name: 自选股列表名称
            stocks: 股票代码列表
        """
        if stocks is None:
            stocks = []

        if name not in self.watchlists:
            self.watchlists[name] = stocks
            self.save_watchlists()

    def delete_watchlist(self, name: str):
        """
        删除自选股列表

        Args:
            name: 自选股列表名称
        """
        if name in self.watchlists and name != "default":
            del self.watchlists[name]
            self.save_watchlists()

    def analyze_watchlist_batch(self, analyzer_func, strategy_name: str = "ma_crossover",
                                watchlist_name: str = "default") -> pd.DataFrame:
        """
        批量分析自选股列表中的股票

        Args:
            analyzer_func: 分析函数，接收股票代码和策略名作为参数
            strategy_name: 分析策略名称
            watchlist_name: 自选股列表名称

        Returns:
            分析结果DataFrame
        """
        watchlist = self.get_watchlist(watchlist_name)
        results = []

        for stock_code in watchlist:
            try:
                result = analyzer_func(stock_code, strategy_name)
                if result is not None:
                    result['stock_code'] = stock_code
                    results.append(result)
            except Exception as e:
                print(f"⚠️ 分析股票 {stock_code} 时出错: {e}")
                continue

        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()


def create_default_watchlist_manager():
    """
    工厂函数：创建默认的自选股管理器
    """
    return WatchlistManager()


if __name__ == "__main__":
    # 示例用法
    wm = WatchlistManager()

    # 添加股票到默认自选股
    wm.add_stock_to_watchlist("000001.SZ")
    wm.add_stock_to_watchlist("600000.SH")
    wm.add_stock_to_watchlist("000002.SZ")

    print("当前自选股列表:", wm.get_watchlist())

    # 创建另一个自选股列表
    wm.create_watchlist("科技股", ["000001.SZ", "300750.SZ", "002475.SZ"])
    print("科技股列表:", wm.get_watchlist("科技股"))

    # 获取所有列表名称
    print("所有自选股列表:", wm.get_watchlist_names())