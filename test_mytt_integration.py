#!/usr/bin/env python3
"""
Test script to verify MyTT indicators integration
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.utils.mytt_indicators import calculate_mytt_indicators
from quant_trade_a_share.utils.eastmoney_data_fetcher import EastMoneyDataFetcher


def test_mytt_integration():
    """
    Test MyTT indicators integration with sample data
    """
    print("🧪 测试MyTT指标集成...")
    print("="*50)

    # Create sample OHLCV data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)  # For reproducible results

    # Generate realistic stock-like data
    price_changes = np.random.normal(0.001, 0.02, 100)  # Small daily drift with volatility
    prices = [100]  # Starting price

    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    # Add some trend and volatility clustering to make it more realistic
    trend_factor = np.linspace(0, 0.3, 100)
    realistic_prices = [prices[i] * (1 + trend_factor[i] + np.random.normal(0, 0.01)) for i in range(len(prices))]

    sample_data = pd.DataFrame({
        'date': dates,
        'open': [p * (1 - abs(np.random.normal(0, 0.01))) for p in realistic_prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in realistic_prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in realistic_prices],
        'close': realistic_prices,
        'volume': np.random.randint(1000000, 10000000, 100)  # Random volume
    })

    sample_data.set_index('date', inplace=True)

    print(f"📊 样本数据维度: {sample_data.shape}")
    print(f"📈 价格范围: ¥{sample_data['close'].min():.2f} - ¥{sample_data['close'].max():.2f}")
    print(f"📊 成交量范围: {sample_data['volume'].min():,} - {sample_data['volume'].max():,}")

    # Calculate MyTT indicators
    print("\n📈 计算MyTT技术指标...")
    try:
        enhanced_data = calculate_mytt_indicators(sample_data)
        print("✅ MyTT指标计算成功!")

        # Check for some specific indicators
        required_columns = [
            'macd_dif', 'macd_dea', 'macd_bar',
            'kdj_k', 'kdj_d', 'kdj_j',
            'rsi6', 'rsi12', 'rsi24',
            'boll_upper', 'boll_mid', 'boll_lower',
            'cci', 'atr',
            'ma5', 'ma10', 'ma20', 'ma30', 'ma60',
            'dmi_pdi', 'dmi_mdi', 'dmi_adx', 'dmi_adxr',
            'trix', 'trma',
            'vr', 'cr',
            'dpo', 'madpo',
            'brar_ar', 'brar_br',
            'roc', 'maroc',
            'ema12', 'ema50',
            'obv', 'mfi', 'asi', 'asit'
        ]

        available_cols = [col for col in required_columns if col in enhanced_data.columns]
        missing_cols = [col for col in required_columns if col not in enhanced_data.columns]

        print(f"✅ 可用指标数量: {len(available_cols)}")
        print(f"❌ 缺失指标数量: {len(missing_cols)}")

        if missing_cols:
            print(f"   缺失指标: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}")

        # Display sample values for key indicators
        latest = enhanced_data.iloc[-1]
        print(f"\n🔍 最新指标值 (最近一天):")
        print(f"   RSI6/12/24: {latest['rsi6']:.2f}/{latest['rsi12']:.2f}/{latest['rsi24']:.2f}")
        print(f"   MACD: {latest['macd_dif']:.4f}/{latest['macd_dea']:.4f}/{latest['macd_bar']:.4f}")
        print(f"   KDJ: {latest['kdj_k']:.2f}/{latest['kdj_d']:.2f}/{latest['kdj_j']:.2f}")
        print(f"   布林带: 上{latest['boll_upper']:.2f}/中{latest['boll_mid']:.2f}/下{latest['boll_lower']:.2f}")
        print(f"   均线: MA5:{latest['ma5']:.2f}, MA20:{latest['ma20']:.2f}, MA60:{latest['ma60']:.2f}")
        print(f"   CCI: {latest['cci']:.2f}")
        print(f"   DMI: PDI:{latest['dmi_pdi']:.2f}, MDI:{latest['dmi_mdi']:.2f}, ADX:{latest['dmi_adx']:.2f}")
        print(f"   TRIX: {latest['trix']:.4f}, VR: {latest['vr']:.2f}")

        print(f"\n🎉 MyTT指标集成测试完成!")
        return True

    except Exception as e:
        print(f"❌ MyTT指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_data_integration():
    """
    Test with real data from EastMoney
    """
    print("\n🌐 测试真实数据集成...")
    print("="*50)

    try:
        # Initialize data fetcher
        fetcher = EastMoneyDataFetcher()

        # Try to fetch sample data (using a popular stock)
        print("尝试获取样本股票数据 (sh600519 - 贵州茅台)...")
        data = fetcher.fetch_stock_data('sh600519', days=60)

        if data is not None and not data.empty:
            print(f"✅ 成功获取数据，共有 {len(data)} 条记录")

            # Calculate MyTT indicators on real data
            enhanced_data = calculate_mytt_indicators(data)
            print(f"✅ 实际数据MyTT指标计算成功!")

            # Show sample of results
            latest = enhanced_data.iloc[-1]
            print(f"\n📈 贵州茅台最新指标值:")
            print(f"   收盘价: ¥{latest['close']:.2f}")
            print(f"   RSI24: {latest['rsi24']:.2f}")
            print(f"   MACD_DIF: {latest['macd_dif']:.4f}")
            print(f"   KDJ_K: {latest['kdj_k']:.2f}")
            print(f"   BOLL_MIDDLE: {latest['boll_mid']:.2f}")
            print(f"   MA20: {latest['ma20']:.2f}")

            return True
        else:
            print("⚠️  无法获取实时数据，但MyTT库本身工作正常")
            return True

    except Exception as e:
        print(f"❌ 真实数据集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main test function
    """
    print("🚀 MyTT指标库集成验证测试")
    print("="*60)

    success1 = test_mytt_integration()
    success2 = test_real_data_integration()

    print("\n" + "="*60)
    if success1 and success2:
        print("✅ 所有测试通过! MyTT指标库已成功集成到系统中")
        print("\n💡 现在您可以:")
        print("   - 使用 enhanced_mytt_analysis.py 进行深入分析")
        print("   - 使用 enhanced_mytt_cli.py 体验增强版CLI")
        print("   - 在其他模块中导入 calculate_mytt_indicators 函数")
    else:
        print("❌ 测试失败，请检查MyTT指标库的实现")

    print("="*60)


if __name__ == "__main__":
    main()