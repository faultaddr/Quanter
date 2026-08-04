"""
基本面数据获取器

数据源优先级：
1. 东方财富 datacenter API（最稳定、数据最全）
2. 东方财富 push2 API（实时估值指标）
3. EnhancedDataFetcher.get_fundamental_data（BaoStock/AkShare 回退）
"""
import re
import requests
from typing import Dict, List, Optional, Any

from ...core.logging import get_logger

logger = get_logger(__name__)

class FundamentalDataProvider:
    """基本面数据获取器（东方财富 API 优先）"""

    # 东方财富 datacenter API
    DATACENTER_URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
    # 东方财富实时行情 API
    QUOTE_URL = 'https://push2.eastmoney.com/api/qt/stock/get'

    def __init__(self):
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        })
        self._no_proxy = {'http': None, 'https': None}

    def get_fundamental_summary(self, symbol: str) -> Dict[str, Any]:
        """
        获取基本面数据摘要

        Args:
            symbol: 股票代码（如 603039、000001.SZ、sz000001）

        Returns:
            Dict: 基本面数据字典，字段与 FundamentalData 对应
        """
        pure_code, market_code = self._normalize_symbol(symbol)

        result = {
            'pe_ttm': None, 'pb': None, 'ps_ttm': None,
            'total_market_cap': None, 'float_market_cap': None,
            'roe': None, 'profit_margin': None, 'gross_margin': None,
            'revenue_yoy': None, 'profit_yoy': None,
            'annual_revenue': None, 'annual_profit': None,
            'eps': None, 'deduct_eps': None, 'debt_ratio': None,
            'annual_history': [],
            'data_source': '',
        }

        # 1. 获取估值数据（PE/PB/市值/毛利率/净利率/ROE/负债率）
        try:
            valuation = self._fetch_valuation(market_code, pure_code)
            if valuation:
                result.update(valuation)
                result['data_source'] = 'eastmoney'
        except Exception as e:
            logger.warning(f"东方财富估值数据获取失败 {symbol}: {e}")

        # 2. 获取财务报表数据（营收/净利/EPS/ROE/历史对比）
        try:
            financials = self._fetch_financial_statements(pure_code)
            if financials:
                for k, v in financials.items():
                    if v is not None:
                        result[k] = v
                if not result['data_source']:
                    result['data_source'] = 'eastmoney'
        except Exception as e:
            logger.warning(f"东方财富财务数据获取失败 {symbol}: {e}")

        # 3. 如果东方财富全部失败，回退到 BaoStock/AkShare
        if not result['data_source']:
            try:
                fallback = self._fetch_from_fallback(symbol)
                if fallback:
                    result.update(fallback)
                    result['data_source'] = 'baostock/akshare'
            except Exception as e:
                logger.warning(f"回退数据源获取失败 {symbol}: {e}")

        return result

    def _normalize_symbol(self, symbol: str) -> tuple:
        """
        标准化股票代码

        Returns:
            (pure_code, market_code): 如 ('603039', '1.603039')
        """
        match = re.search(r'(\d{6})', symbol)
        if not match:
            return symbol, f'1.{symbol}'
        pure_code = match.group(1)
        # 上证: 1.xxxxxx, 深证: 0.xxxxxx
        if pure_code.startswith(('6', '5', '9')):
            market_code = f'1.{pure_code}'
        else:
            market_code = f'0.{pure_code}'
        return pure_code, market_code

    def _fetch_valuation(self, market_code: str, pure_code: str) -> Optional[Dict]:
        """从东方财富获取实时估值指标"""
        params = {
            'secid': market_code,
            'fields': 'f57,f58,f162,f163,f164,f167,f168,f169,f170,f171,f185,f186,f187,f188,f190',
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        }
        r = self._session.get(
            self.QUOTE_URL, params=params,
            timeout=10, proxies=self._no_proxy
        )
        data = r.json()
        if not data.get('data'):
            return None

        d = data['data']
        # 字段映射:
        # f162=PE(动), f163=PE(静), f164=PE(TTM)
        # f167=PB, f168=总市值(万), f169=流通市值(万)
        # f185=ROE(加权%), f186=毛利率%, f187=净利率%
        # f188=营收同比%, f190=负债率%
        result = {}
        result['pe_ttm'] = self._safe_div(d.get('f164'))
        result['pb'] = self._safe_div(d.get('f167'))
        total_mv = d.get('f168')
        float_mv = d.get('f169')
        if total_mv and total_mv > 0:
            result['total_market_cap'] = round(total_mv / 10000, 2)  # 万→亿
        if float_mv and float_mv > 0:
            result['float_market_cap'] = round(float_mv / 10000, 2)
        result['roe'] = self._safe_div(d.get('f185'))
        result['gross_margin'] = self._safe_div(d.get('f186'))
        result['profit_margin'] = self._safe_div(d.get('f187'))
        result['revenue_yoy'] = self._safe_div(d.get('f188'))
        result['debt_ratio'] = self._safe_div(d.get('f190'))

        return result

    def _fetch_financial_statements(self, pure_code: str) -> Optional[Dict]:
        """从东方财富 datacenter 获取财务报表数据"""
        result = {}

        # 年报数据
        annual = self._fetch_report_data(pure_code, '年报', 8)
        if annual:
            # 按报告期降序排列
            annual.sort(key=lambda x: x.get('QDATE', ''), reverse=True)

            # 最新年报
            latest = annual[0]
            result['annual_revenue'] = self._to_yi(latest.get('TOTAL_OPERATE_INCOME'))
            result['annual_profit'] = self._to_yi(latest.get('PARENT_NETPROFIT'))
            result['eps'] = latest.get('BASIC_EPS')
            result['deduct_eps'] = latest.get('DEDUCT_BASIC_EPS')
            result['roe'] = latest.get('WEIGHTAVG_ROE') or result.get('roe')

            # 历史对比
            history = []
            for item in annual[:5]:
                history.append({
                    'year': item.get('QDATE', '')[:4],
                    'revenue': self._to_yi(item.get('TOTAL_OPERATE_INCOME')),
                    'profit': self._to_yi(item.get('PARENT_NETPROFIT')),
                    'eps': item.get('BASIC_EPS'),
                    'roe': item.get('WEIGHTAVG_ROE'),
                    'deduct_eps': item.get('DEDUCT_BASIC_EPS'),
                })
            result['annual_history'] = history

            # 计算同比（取前两年对比）
            if len(annual) >= 2:
                curr_rev = annual[0].get('TOTAL_OPERATE_INCOME', 0) or 0
                prev_rev = annual[1].get('TOTAL_OPERATE_INCOME', 0) or 0
                if prev_rev > 0:
                    result['revenue_yoy'] = round((curr_rev - prev_rev) / prev_rev * 100, 1)

                curr_prof = annual[0].get('PARENT_NETPROFIT', 0) or 0
                prev_prof = annual[1].get('PARENT_NETPROFIT', 0) or 0
                if prev_prof > 0:
                    result['profit_yoy'] = round((curr_prof - prev_prof) / prev_prof * 100, 1)

        return result

    def _fetch_report_data(
        self, pure_code: str, report_type: str, count: int = 8
    ) -> List[Dict]:
        """获取指定类型的财务报表数据"""
        params = {
            'reportName': 'RPT_LICO_FN_CPD',
            'columns': 'ALL',
            'filter': f'(SECURITY_CODE="{pure_code}")(DATEMMDD="{report_type}")',
            'pageNumber': '1',
            'pageSize': str(count),
            'source': 'WEB',
            'client': 'WEB',
        }
        r = self._session.get(
            self.DATACENTER_URL, params=params,
            timeout=10, proxies=self._no_proxy
        )
        data = r.json()
        if data.get('result') and data['result'].get('data'):
            return data['result']['data']
        return []

    def _fetch_from_fallback(self, symbol: str) -> Optional[Dict]:
        """回退到 BaoStock/AkShare 获取基本面数据"""
        try:
            from .historical.enhanced_fetcher import EnhancedDataFetcher
            fetcher = EnhancedDataFetcher()
            data = fetcher.get_fundamental_data(symbol)
            if data and not data.get('error'):
                result = {}
                result['pe_ttm'] = data.get('pe')
                result['pb'] = data.get('pb')
                result['roe'] = data.get('roe')
                result['profit_margin'] = data.get('profit_margin')
                result['eps'] = data.get('eps')
                result['revenue_yoy'] = data.get('yoy_profit')
                return result
        except Exception as e:
            logger.warning(f"回退数据源获取失败: {e}")
        return None

    @staticmethod
    def _safe_div(val) -> Optional[float]:
        """安全转换数值（东方财富返回放大100倍的整数）"""
        if val is None:
            return None
        try:
            v = float(val)
            # 东方财富百分比如 ROE/毛利率/净利率/负债率 放大100倍
            # PE/PB 等估值指标也是放大100倍
            return round(v / 100, 2)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_yi(val) -> Optional[float]:
        """转换为亿元"""
        if val is None:
            return None
        try:
            return round(float(val) / 1e8, 2)
        except (ValueError, TypeError):
            return None
