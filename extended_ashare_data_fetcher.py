"""
扩展的Ashare数据获取器
此模块扩展了原有的Ashare数据获取功能，提供更多的基本面数据获取选项
"""

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from quant_trade_a_share.data.ashare_data_fetcher import AshareDataFetcher


class ExtendedAshareDataFetcher(AshareDataFetcher):
    """
    扩展的Ashare数据获取器，增加了基本面数据获取功能
    """

    def __init__(self, tushare_token=None):
        """
        初始化扩展的Ashare数据获取器
        """
        super().__init__()
        self.tushare_token = tushare_token
        if tushare_token:
            ts.set_token(tushare_token)
            self.pro = ts.pro_api()
            print("✅ Tushare API已初始化，可获取基本面数据")
        else:
            print("⚠️ 未提供Tushare Token，只能获取基本价格数据")
            self.pro = None

    def _convert_symbol_for_tushare(self, symbol):
        """
        将Ashare符号转换为Tushare格式
        """
        # 移除可能的前缀并转换为Tushare格式
        if symbol.startswith('sh') or symbol.startswith('sz'):
            code = symbol[2:]
            if symbol.startswith('sh'):
                if code.startswith(('5', '6')):  # 上交所股票
                    return f"{code}.SH"
                else:  # ETF基金
                    return f"{code}.SH"
            elif symbol.startswith('sz'):
                return f"{code}.SZ"
        return symbol

    def fetch_basic_info(self, symbol):
        """
        获取股票基本信息
        """
        if not self.pro:
            print("❌ Tushare Token未设置，无法获取基本信息")
            return None

        try:
            symbol_ts = self._convert_symbol_for_tushare(symbol)
            # 使用tushare获取股票基本信息
            df = self.pro.stock_basic(ts_code=symbol_ts, fields='ts_code,symbol,name,area,industry,fullname,enname,market,exchange,curr_type,list_status,list_date,delist_date,is_hs')
            return df
        except Exception as e:
            print(f"❌ 获取 {symbol} 基本信息失败: {e}")
            return None

    def fetch_balance_sheet(self, symbol, period=None):
        """
        获取资产负债表

        Args:
            symbol: 股票代码
            period: 报告期(YYYYQ)，如2023Q1, 2022Q4等
        """
        if not self.pro:
            print("❌ Tushare Token未设置，无法获取资产负债表")
            return None

        try:
            symbol_ts = self._convert_symbol_for_tushare(symbol)

            # 如果未指定period，则获取最近4个季度的数据
            if period is None:
                # 获取最近的4个季度
                current_year = datetime.now().year
                current_quarter = (datetime.now().month - 1) // 3 + 1

                # 获取最近4个季度的报告期
                periods = []
                for i in range(4):
                    q = current_quarter - i
                    y = current_year
                    while q <= 0:
                        q += 4
                        y -= 1
                    periods.append(f"{y}{q:02d}")

                df_list = []
                for p in periods:
                    try:
                        df = self.pro.balancesheet_vip(ts_code=symbol_ts, period=p)
                        if df is not None and not df.empty:
                            df_list.append(df)
                    except:
                        continue

                if df_list:
                    return pd.concat(df_list, ignore_index=True)
                else:
                    return pd.DataFrame()
            else:
                df = self.pro.balancesheet_vip(ts_code=symbol_ts, period=period)
                return df

        except Exception as e:
            print(f"❌ 获取 {symbol} 资产负债表失败: {e}")
            return None

    def fetch_income_statement(self, symbol, period=None):
        """
        获取利润表

        Args:
            symbol: 股票代码
            period: 报告期(YYYYQ)
        """
        if not self.pro:
            print("❌ Tushare Token未设置，无法获取利润表")
            return None

        try:
            symbol_ts = self._convert_symbol_for_tushare(symbol)

            # 如果未指定period，则获取最近4个季度的数据
            if period is None:
                # 获取最近的4个季度
                current_year = datetime.now().year
                current_quarter = (datetime.now().month - 1) // 3 + 1

                periods = []
                for i in range(4):
                    q = current_quarter - i
                    y = current_year
                    while q <= 0:
                        q += 4
                        y -= 1
                    periods.append(f"{y}{q:02d}")

                df_list = []
                for p in periods:
                    try:
                        df = self.pro.income_vip(ts_code=symbol_ts, period=p)
                        if df is not None and not df.empty:
                            df_list.append(df)
                    except:
                        continue

                if df_list:
                    return pd.concat(df_list, ignore_index=True)
                else:
                    return pd.DataFrame()
            else:
                df = self.pro.income_vip(ts_code=symbol_ts, period=period)
                return df

        except Exception as e:
            print(f"❌ 获取 {symbol} 利润表失败: {e}")
            return None

    def fetch_cash_flow(self, symbol, period=None):
        """
        获取现金流量表

        Args:
            symbol: 股票代码
            period: 报告期(YYYYQ)
        """
        if not self.pro:
            print("❌ Tushare Token未设置，无法获取现金流量表")
            return None

        try:
            symbol_ts = self._convert_symbol_for_tushare(symbol)

            # 如果未指定period，则获取最近4个季度的数据
            if period is None:
                # 获取最近的4个季度
                current_year = datetime.now().year
                current_quarter = (datetime.now().month - 1) // 3 + 1

                periods = []
                for i in range(4):
                    q = current_quarter - i
                    y = current_year
                    while q <= 0:
                        q += 4
                        y -= 1
                    periods.append(f"{y}{q:02d}")

                df_list = []
                for p in periods:
                    try:
                        df = self.pro.cashflow_vip(ts_code=symbol_ts, period=p)
                        if df is not None and not df.empty:
                            df_list.append(df)
                    except:
                        continue

                if df_list:
                    return pd.concat(df_list, ignore_index=True)
                else:
                    return pd.DataFrame()
            else:
                df = self.pro.cashflow_vip(ts_code=symbol_ts, period=period)
                return df

        except Exception as e:
            print(f"❌ 获取 {symbol} 现金流量表失败: {e}")
            return None

    def fetch_financial_indicator(self, symbol, period=None):
        """
        获取财务指标数据

        Args:
            symbol: 股票代码
            period: 报告期(YYYYQ)
        """
        if not self.pro:
            print("❌ Tushare Token未设置，无法获取财务指标")
            return None

        try:
            symbol_ts = self._convert_symbol_for_tushare(symbol)

            # 如果未指定period，则获取最近4个季度的数据
            if period is None:
                # 获取最近的4个季度
                current_year = datetime.now().year
                current_quarter = (datetime.now().month - 1) // 3 + 1

                periods = []
                for i in range(4):
                    q = current_quarter - i
                    y = current_year
                    while q <= 0:
                        q += 4
                        y -= 1
                    periods.append(f"{y}{q:02d}")

                df_list = []
                for p in periods:
                    try:
                        df = self.pro.fina_indicator_vip(ts_code=symbol_ts, period=p)
                        if df is not None and not df.empty:
                            df_list.append(df)
                    except:
                        continue

                if df_list:
                    return pd.concat(df_list, ignore_index=True)
                else:
                    return pd.DataFrame()
            else:
                df = self.pro.fina_indicator_vip(ts_code=symbol_ts, period=period)
                return df

        except Exception as e:
            print(f"❌ 获取 {symbol} 财务指标失败: {e}")
            return None

    def fetch_main_operation_data(self, symbol, period=None):
        """
        获取主营业务数据

        Args:
            symbol: 股票代码
            period: 报告期(YYYYQ)
        """
        if not self.pro:
            print("❌ Tushare Token未设置，无法获取主营业务数据")
            return None

        try:
            symbol_ts = self._convert_symbol_for_tushare(symbol)

            # 如果未指定period，则获取最近4个季度的数据
            if period is None:
                # 获取最近的4个季度
                current_year = datetime.now().year
                current_quarter = (datetime.now().month - 1) // 3 + 1

                periods = []
                for i in range(4):
                    q = current_quarter - i
                    y = current_year
                    while q <= 0:
                        q += 4
                        y -= 1
                    periods.append(f"{y}{q:02d}")

                df_list = []
                for p in periods:
                    try:
                        df = self.pro.fina_mainbz_vip(ts_code=symbol_ts, period=p)
                        if df is not None and not df.empty:
                            df_list.append(df)
                    except:
                        continue

                if df_list:
                    return pd.concat(df_list, ignore_index=True)
                else:
                    return pd.DataFrame()
            else:
                df = self.pro.fina_mainbz_vip(ts_code=symbol_ts, period=period)
                return df

        except Exception as e:
            print(f"❌ 获取 {symbol} 主营业务数据失败: {e}")
            return None

    def fetch_announcements(self, symbol, date=None):
        """
        获取公司公告（如果可用）

        Args:
            symbol: 股票代码
            date: 查询日期(YYYY-MM-DD)
        """
        if not self.pro:
            print("❌ Tushare Token未设置，无法获取公告")
            return None

        try:
            symbol_ts = self._convert_symbol_for_tushare(symbol)

            # 获取最近的公告
            if date is None:
                # 获取最近一个月的公告
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                df = self.pro.disclosure_info(ts_code=symbol_ts, start_date=start_date, end_date=end_date)
            else:
                date_formatted = date.replace('-', '')
                df = self.pro.disclosure_info(ts_code=symbol_ts, date=date_formatted)

            return df
        except Exception as e:
            print(f"❌ 获取 {symbol} 公告失败: {e}")
            return None

    def fetch_comprehensive_fundamental_data(self, symbol):
        """
        获取综合基本面数据（基本信息+财务报表+财务指标）
        """
        print(f"🔍 正在获取 {symbol} 的综合基本面数据...")

        fundamental_data = {}

        # 1. 获取基本信息
        print("📊 获取基本信息...")
        basic_info = self.fetch_basic_info(symbol)
        if basic_info is not None and not basic_info.empty:
            fundamental_data['basic_info'] = basic_info.iloc[0] if len(basic_info) > 0 else None

        # 2. 获取财务指标
        print("📈 获取财务指标...")
        financial_indicators = self.fetch_financial_indicator(symbol)
        if financial_indicators is not None and not financial_indicators.empty:
            fundamental_data['financial_indicators'] = financial_indicators

        # 3. 获取资产负债表
        print("🏢 获取资产负债表...")
        balance_sheet = self.fetch_balance_sheet(symbol)
        if balance_sheet is not None and not balance_sheet.empty:
            fundamental_data['balance_sheet'] = balance_sheet

        # 4. 获取利润表
        print("💰 获取利润表...")
        income_statement = self.fetch_income_statement(symbol)
        if income_statement is not None and not income_statement.empty:
            fundamental_data['income_statement'] = income_statement

        # 5. 获取现金流量表
        print("💵 获取现金流量表...")
        cash_flow = self.fetch_cash_flow(symbol)
        if cash_flow is not None and not cash_flow.empty:
            fundamental_data['cash_flow'] = cash_flow

        # 6. 获取主营业务数据
        print("🏭 获取主营业务数据...")
        main_operation = self.fetch_main_operation_data(symbol)
        if main_operation is not None and not main_operation.empty:
            fundamental_data['main_operation'] = main_operation

        print(f"✅ {symbol} 的综合基本面数据获取完成!")
        return fundamental_data


def test_extended_fetcher():
    """
    测试扩展的Ashare数据获取器
    """
    print("🧪 测试扩展Ashare数据获取器...")

    # 初始化获取器（使用默认token）
    token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"
    extended_fetcher = ExtendedAshareDataFetcher(tushare_token=token)

    # 测试股票
    symbol = "sh600023"  # 华能国际

    print(f"\n🔍 获取 {symbol} 的基本面数据...")

    # 获取综合基本面数据
    fundamental_data = extended_fetcher.fetch_comprehensive_fundamental_data(symbol)

    if fundamental_data:
        print(f"\n📊 获取的数据类型:")
        for key in fundamental_data.keys():
            if fundamental_data[key] is not None:
                if isinstance(fundamental_data[key], pd.DataFrame):
                    print(f"  - {key}: {len(fundamental_data[key])} 条记录")
                else:
                    print(f"  - {key}: 单条记录")

        # 显示关键财务指标的摘要
        if 'financial_indicators' in fundamental_data and fundamental_data['financial_indicators'] is not None:
            indicators = fundamental_data['financial_indicators']
            if not indicators.empty:
                print(f"\n📈 最新财务指标摘要:")
                latest_indicators = indicators.iloc[0] if len(indicators) > 0 else None
                if latest_indicators is not None:
                    print(f"  - ROE (净资产收益率): {latest_indicators.get('roe', 'N/A')}")
                    print(f"  - ROA (总资产报酬率): {latest_indicators.get('roa', 'N/A')}")
                    print(f"  - Debt-to-Asset Ratio (资产负债率): {latest_indicators.get('debt_to_assets', 'N/A')}")
                    print(f"  - Gross Profit Margin (销售毛利率): {latest_indicators.get('gross_profit_margin', 'N/A')}")
                    print(f"  - Current Ratio (流动比率): {latest_indicators.get('current_ratio', 'N/A')}")

    return extended_fetcher, fundamental_data


if __name__ == "__main__":
    fetcher, data = test_extended_fetcher()