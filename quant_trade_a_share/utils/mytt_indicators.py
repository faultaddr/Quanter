"""
MyTT Technical Indicators Integration
This module integrates the MyTT technical indicators library into our quant trading system.
Based on: https://github.com/mpquant/MyTT
"""

import numpy as np
import pandas as pd
import math


#------------------ 0级：核心工具函数 --------------------------------------------
def RD(N, D=3):
    """四舍五入取D位小数，默认3位"""
    return np.round(N, D)


def RET(S, N=1):
    """返回序列倒数第N个值,默认返回最后一个"""
    return np.array(S)[-N]


def ABS(S):
    """返回N的绝对值"""
    return np.abs(S)


def LN(S):
    """求底是e的自然对数"""
    return np.log(S)


def POW(S, N):
    """求S的N次方"""
    return np.power(S, N)


def SQRT(S):
    """求S的平方根"""
    return np.sqrt(S)


def SIN(S):
    """求S的正弦值（弧度)"""
    return np.sin(S)


def COS(S):
    """求S的余弦值（弧度)"""
    return np.cos(S)


def TAN(S):
    """求S的正切值（弧度)"""
    return np.tan(S)


def MAX(S1, S2):
    """序列max"""
    return np.maximum(S1, S2)


def MIN(S1, S2):
    """序列min"""
    return np.minimum(S1, S2)


def IF(S, A, B):
    """序列布尔判断 return=A if S==True else B"""
    return np.where(S, A, B)


def REF(S, N=1):
    """对序列整体下移动N,返回序列(shift后会产生NAN)"""
    return pd.Series(S).shift(N).values


def DIFF(S, N=1):
    """前一个值减后一个值,前面会产生nan"""
    return pd.Series(S).diff(N).values


def STD(S, N):
    """求序列的N日标准差，返回序列"""
    return pd.Series(S).rolling(N).std(ddof=0).values


def SUM(S, N):
    """对序列求N天累计和，返回序列 N=0对序列所有依次求和"""
    return pd.Series(S).rolling(N).sum().values if N > 0 else pd.Series(S).cumsum().values


def CONST(S):
    """返回序列S最后的值组成常量序列"""
    return np.full(len(S), S[-1])


def HHV(S, N):
    """HHV(C, 5) 最近5天收盘最高价"""
    return pd.Series(S).rolling(N).max().values


def LLV(S, N):
    """LLV(C, 5) 最近5天收盘最低价"""
    return pd.Series(S).rolling(N).min().values


def HHVBARS(S, N):
    """求N周期内S最高值到当前周期数, 返回序列"""
    return pd.Series(S).rolling(N).apply(lambda x: np.argmax(x[::-1]), raw=True).values


def LLVBARS(S, N):
    """求N周期内S最低值到当前周期数, 返回序列"""
    return pd.Series(S).rolling(N).apply(lambda x: np.argmin(x[::-1]), raw=True).values


def MA(S, N):
    """求序列的N日简单移动平均值，返回序列"""
    return pd.Series(S).rolling(N).mean().values


def EMA(S, N):
    """指数移动平均,为了精度 S>4*N EMA至少需要120周期 alpha=2/(span+1)"""
    return pd.Series(S).ewm(span=N, adjust=False).mean().values


def SMA(S, N, M=1):
    """中国式的SMA,至少需要120周期才精确 (雪球180周期) alpha=1/(1+com)"""
    return pd.Series(S).ewm(alpha=M/N, adjust=False).mean().values


def WMA(S, N):
    """通达信S序列的N日加权移动平均 Yn = (1*X1+2*X2+3*X3+...+n*Xn)/(1+2+3+...+Xn)"""
    return pd.Series(S).rolling(N).apply(lambda x: x[::-1].cumsum().sum()*2/N/(N+1), raw=True).values


def DMA(S, A):
    """求S的动态移动平均，A作平滑因子,必须 0<A<1 (此为核心函数，非指标）"""
    if isinstance(A, (int, float)):
        return pd.Series(S).ewm(alpha=A, adjust=False).mean().values
    A = np.array(A)
    A[np.isnan(A)] = 1.0
    Y = np.zeros(len(S))
    Y[0] = S[0]
    for i in range(1, len(S)):
        Y[i] = A[i]*S[i] + (1-A[i])*Y[i-1]  # A支持序列 by jqz1226
    return Y


def AVEDEV(S, N):
    """平均绝对偏差 (序列与其平均值的绝对差的平均值)"""
    return pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean()).values


def SLOPE(S, N):
    """返S序列N周期回线性回归斜率"""
    return pd.Series(S).rolling(N).apply(lambda x: np.polyfit(range(N), x, deg=1)[0], raw=True).values


def FORCAST(S, N):
    """返回S序列N周期回线性回归后的预测值， jqz1226改进成序列出"""
    return pd.Series(S).rolling(N).apply(lambda x: np.polyval(np.polyfit(range(N), x, deg=1), N-1), raw=True).values


def LAST(S, A, B):
    """从前A日到前B日一直满足S_BOOL条件, 要求A>B & A>0 & B>=0"""
    return np.array(pd.Series(S).rolling(A+1).apply(lambda x: np.all(x[::-1][B:]), raw=True), dtype=bool)


#------------------ 1级：应用层函数(通过0级核心函数实现）--------------------------------
def COUNT(S, N):
    """COUNT(CLOSE>O, N): 最近N天满足S_BOO的天数 True的天数"""
    return SUM(S, N)


def EVERY(S, N):
    """EVERY(CLOSE>O, 5) 最近N天是否都是True"""
    return IF(SUM(S, N) == N, True, False)


def EXIST(S, N):
    """EXIST(CLOSE>3010, N=5) n日内是否存在一天大于3000点"""
    return IF(SUM(S, N) > 0, True, False)


def FILTER(S, N):
    """FILTER函数，S满足条件后，将其后N周期内的数据置为0, FILTER(C==H,5)"""
    S = S.copy()  # 创建副本避免修改原数据
    for i in range(len(S)):
        if S[i]:
            start_idx = min(i + 1, len(S))
            end_idx = min(i + 1 + N, len(S))
            S[start_idx:end_idx] = 0
    return S


def BARSLAST(S):
    """上一次条件成立到当前的周期, BARSLAST(C/REF(C,1)>=1.1) 上一次涨停到今天的天数"""
    M = np.concatenate(([0], np.where(S, 1, 0)))
    for i in range(1, len(M)):
        M[i] = 0 if M[i] else M[i-1] + 1
    return M[1:]


def BARSLASTCOUNT(S):
    """统计连续满足S条件的周期数"""
    rt = np.zeros(len(S)+1)  # BARSLASTCOUNT(CLOSE>OPEN)表示统计连续收阳的周期数
    for i in range(len(S)):
        rt[i+1] = rt[i] + 1 if S[i] else rt[i+1]
    return rt[1:]


def BARSSINCEN(S, N):
    """N周期内第一次S条件成立到现在的周期数,N为常量 by jqz1226"""
    return pd.Series(S).rolling(N).apply(lambda x: N-1-np.argmax(x) if np.argmax(x) or x[0] else 0, raw=True).fillna(0).values.astype(int)


def CROSS(S1, S2):
    """判断向上金叉穿越 CROSS(MA(C,5),MA(C,10)) 判断向下死叉穿越 CROSS(MA(C,10),MA(C,5))"""
    return np.concatenate(([False], np.logical_not((S1 > S2)[:-1]) & (S1 > S2)[1:]))  # 不使用0级函数,移植方便 by jqz1226


def LONGCROSS(S1, S2, N):
    """两条线维持一定周期后交叉,S1在N周期内都小于S2,本周期从S1下方向上穿过S2时返回1,否则返回0"""
    return np.array(np.logical_and(LAST(S1 < S2, N, 1), (S1 > S2)), dtype=bool)  # N=1时等同于CROSS(S1, S2)


def VALUEWHEN(S, X):
    """当S条件成立时,取X的当前值,否则取VALUEWHEN的上个成立时的X值 by jqz1226"""
    return pd.Series(np.where(S, X, np.nan)).ffill().values


def BETWEEN(S, A, B):
    """S处于A和B之间时为真。 包括 A<S<B 或 A>S>B"""
    return ((A < S) & (S < B)) | ((A > S) & (S > B))


def TOPRANGE(S):
    """TOPRANGE(HIGH)表示当前最高价是近多少周期内最高价的最大值 by jqz1226"""
    rt = np.zeros(len(S))
    for i in range(1, len(S)):
        rt[i] = np.argmin(np.flipud(S[:i] < S[i]))
    return rt.astype('int')


def LOWRANGE(S):
    """LOWRANGE(LOW)表示当前最低价是近多少周期内最低价的最小值 by jqz1226"""
    rt = np.zeros(len(S))
    for i in range(1, len(S)):
        rt[i] = np.argmin(np.flipud(S[:i] > S[i]))
    return rt.astype('int')


#------------------ 2级：技术指标函数(全部通过0级，1级函数实现） ------------------------------
def MACD(CLOSE, SHORT=12, LONG=26, M=9):
    """MACD指标"""
    DIF = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
    DEA = EMA(DIF, M)
    MACD = (DIF - DEA) * 2
    return RD(DIF), RD(DEA), RD(MACD)


def KDJ(CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
    """KDJ指标"""
    RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    K = EMA(RSV, (M1*2-1))
    D = EMA(K, (M2*2-1))
    J = K*3 - D*2
    return K, D, J


def RSI(CLOSE, N=24):
    """RSI指标,和通达信小数点2位相同"""
    DIF = CLOSE - REF(CLOSE, 1)
    return RD(SMA(MAX(DIF, 0), N) / SMA(ABS(DIF), N) * 100)


def WR(CLOSE, HIGH, LOW, N=10, N1=6):
    """W&R 威廉指标"""
    WR = (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    WR1 = (HHV(HIGH, N1) - CLOSE) / (HHV(HIGH, N1) - LLV(LOW, N1)) * 100
    return RD(WR), RD(WR1)


def BIAS(CLOSE, L1=6, L2=12, L3=24):
    """BIAS乖离率"""
    BIAS1 = (CLOSE - MA(CLOSE, L1)) / MA(CLOSE, L1) * 100
    BIAS2 = (CLOSE - MA(CLOSE, L2)) / MA(CLOSE, L2) * 100
    BIAS3 = (CLOSE - MA(CLOSE, L3)) / MA(CLOSE, L3) * 100
    return RD(BIAS1), RD(BIAS2), RD(BIAS3)


def BOLL(CLOSE, N=20, P=2):
    """BOLL指标，布林带"""
    MID = MA(CLOSE, N)
    UPPER = MID + STD(CLOSE, N) * P
    LOWER = MID - STD(CLOSE, N) * P
    return RD(UPPER), RD(MID), RD(LOWER)


def PSY(CLOSE, N=12, M=6):
    """PSY心理线指标"""
    PSY = COUNT(CLOSE > REF(CLOSE, 1), N) / N * 100
    PSYMA = MA(PSY, M)
    return RD(PSY), RD(PSYMA)


def CCI(CLOSE, HIGH, LOW, N=14):
    """CCI商品路径指标"""
    TP = (HIGH + LOW + CLOSE) / 3
    return (TP - MA(TP, N)) / (0.015 * AVEDEV(TP, N))


def ATR(CLOSE, HIGH, LOW, N=20):
    """真实波动N日平均值"""
    TR = MAX(MAX((HIGH - LOW), ABS(REF(CLOSE, 1) - HIGH)), ABS(REF(CLOSE, 1) - LOW))
    return MA(TR, N)


def BBI(CLOSE, M1=3, M2=6, M3=12, M4=20):
    """BBI多空指标"""
    return (MA(CLOSE, M1) + MA(CLOSE, M2) + MA(CLOSE, M3) + MA(CLOSE, M4)) / 4


def DMI(CLOSE, HIGH, LOW, M1=14, M2=6):
    """动向指标：结果和同花顺，通达信完全一致"""
    TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))), ABS(LOW - REF(CLOSE, 1))), M1)
    HD = HIGH - REF(HIGH, 1)
    LD = REF(LOW, 1) - LOW
    DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), M1)
    DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
    PDI = DMP * 100 / TR
    MDI = DMM * 100 / TR
    ADX = MA(ABS(MDI - PDI) / (PDI + MDI) * 100, M2)
    ADXR = (ADX + REF(ADX, M2)) / 2
    return PDI, MDI, ADX, ADXR


def TAQ(HIGH, LOW, N):
    """唐安奇通道(海龟)交易指标，大道至简，能穿越牛熊"""
    UP = HHV(HIGH, N)
    DOWN = LLV(LOW, N)
    MID = (UP + DOWN) / 2
    return UP, MID, DOWN


def KTN(CLOSE, HIGH, LOW, N=20, M=10):
    """肯特纳交易通道, N选20日，ATR选10日"""
    MID = EMA((HIGH + LOW + CLOSE) / 3, N)
    ATRN = ATR(CLOSE, HIGH, LOW, M)
    UPPER = MID + 2*ATRN
    LOWER = MID - 2*ATRN
    return UPPER, MID, LOWER


def TRIX(CLOSE, M1=12, M2=20):
    """三重指数平滑平均线"""
    TR = EMA(EMA(EMA(CLOSE, M1), M1), M1)
    TRIX = (TR - REF(TR, 1)) / REF(TR, 1) * 100
    TRMA = MA(TRIX, M2)
    return TRIX, TRMA


def VR(CLOSE, VOL, M1=26):
    """VR容量比率"""
    LC = REF(CLOSE, 1)
    return SUM(IF(CLOSE > LC, VOL, 0), M1) / SUM(IF(CLOSE <= LC, VOL, 0), M1) * 100


def CR(CLOSE, HIGH, LOW, N=20):
    """CR价格动量指标"""
    MID = REF(HIGH + LOW + CLOSE, 1) / 3
    return SUM(MAX(0, HIGH - MID), N) / SUM(MAX(0, MID - LOW), N) * 100


def EMV(HIGH, LOW, VOL, N=14, M=9):
    """简易波动指标"""
    VOLUME = MA(VOL, N) / VOL
    MID = 100 * (HIGH + LOW - REF(HIGH + LOW, 1)) / (HIGH + LOW)
    EMV = MA(MID * VOLUME * (HIGH - LOW) / MA(HIGH - LOW, N), N)
    MAEMV = MA(EMV, M)
    return EMV, MAEMV


def DPO(CLOSE, M1=20, M2=10, M3=6):
    """区间震荡线"""
    DPO = CLOSE - REF(MA(CLOSE, M1), M2)
    MADPO = MA(DPO, M3)
    return DPO, MADPO


def BRAR(OPEN, CLOSE, HIGH, LOW, M1=26):
    """BRAR-ARBR 情绪指标"""
    AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
    BR = SUM(MAX(0, HIGH - REF(CLOSE, 1)), M1) / SUM(MAX(0, REF(CLOSE, 1) - LOW), M1) * 100
    return AR, BR


def DFMA(CLOSE, N1=10, N2=50, M=10):
    """平行线差指标"""
    DIF = MA(CLOSE, N1) - MA(CLOSE, N2)
    DIFMA = MA(DIF, M)  # 通达信指标叫DMA 同花顺叫新DMA
    return DIF, DIFMA


def MTM(CLOSE, N=12, M=6):
    """动量指标"""
    MTM = CLOSE - REF(CLOSE, N)
    MTMMA = MA(MTM, M)
    return MTM, MTMMA


def MASS(HIGH, LOW, N1=9, N2=25, M=6):
    """梅斯线"""
    MASS = SUM(MA(HIGH-LOW, N1) / MA(MA(HIGH-LOW, N1), N1), N2)
    MA_MASS = MA(MASS, M)
    return MASS, MA_MASS


def ROC(CLOSE, N=12, M=6):
    """变动率指标"""
    ROC = 100 * (CLOSE - REF(CLOSE, N)) / REF(CLOSE, N)
    MAROC = MA(ROC, M)
    return ROC, MAROC


def EXPMA(CLOSE, N1=12, N2=50):
    """EMA指数平均数指标"""
    return EMA(CLOSE, N1), EMA(CLOSE, N2)


def OBV(CLOSE, VOL):
    """能量潮指标"""
    return SUM(IF(CLOSE > REF(CLOSE, 1), VOL, IF(CLOSE < REF(CLOSE, 1), -VOL, 0)), 0) / 10000


def MFI(CLOSE, HIGH, LOW, VOL, N=14):
    """MFI指标是成交量的RSI指标"""
    TYP = (HIGH + LOW + CLOSE) / 3
    V1 = SUM(IF(TYP > REF(TYP, 1), TYP * VOL, 0), N) / SUM(IF(TYP < REF(TYP, 1), TYP * VOL, 0), N)
    return 100 - (100 / (1 + V1))


def ASI(OPEN, CLOSE, HIGH, LOW, M1=26, M2=10):
    """振动升降指标"""
    LC = REF(CLOSE, 1)
    AA = ABS(HIGH - LC)
    BB = ABS(LOW - LC)
    CC = ABS(HIGH - REF(LOW, 1))
    DD = ABS(LC - REF(OPEN, 1))
    R = IF((AA > BB) & (AA > CC), AA + BB/2 + DD/4, IF((BB > CC) & (BB > AA), BB + AA/2 + DD/4, CC + DD/4))
    X = (CLOSE - LC + (CLOSE - OPEN)/2 + LC - REF(OPEN, 1))
    SI = 16 * X / R * MAX(AA, BB)
    ASI = SUM(SI, M1)
    ASIT = MA(ASI, M2)
    return ASI, ASIT


def XSII(CLOSE, HIGH, LOW, N=102, M=7):
    """薛斯通道II"""
    AA = MA((2*CLOSE + HIGH + LOW)/4, 5)  # 最新版DMA才支持 2021-12-4
    TD1 = AA * N / 100
    TD2 = AA * (200 - N) / 100
    CC = ABS((2*CLOSE + HIGH + LOW)/4 - MA(CLOSE, 20)) / MA(CLOSE, 20)
    DD = DMA(CLOSE, CC)
    TD3 = (1 + M/100) * DD
    TD4 = (1 - M/100) * DD
    return TD1, TD2, TD3, TD4


def calculate_mytt_indicators(df):
    """
    使用MyTT指标库计算所有技术指标
    参数:
    df: 包含OHLCV数据的pandas DataFrame
    返回:
    添加了所有技术指标的DataFrame
    """
    # 确保数据按时间顺序排序
    df = df.sort_index()

    # 转换为numpy数组用于计算
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df)) * 1000
    open_ = df['open'].values if 'open' in df.columns else close  # 如果没有开盘价，用收盘价代替

    # MACD
    dif, dea, bar = MACD(close)
    df['macd_dif'] = dif
    df['macd_dea'] = dea
    df['macd_bar'] = bar

    # KDJ
    k, d, j = KDJ(close, high, low)
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = j

    # RSI
    rsi6 = RSI(close, 6)
    rsi12 = RSI(close, 12)
    rsi24 = RSI(close, 24)
    df['rsi6'] = rsi6
    df['rsi12'] = rsi12
    df['rsi24'] = rsi24

    # 威廉指标
    wr1, wr2 = WR(close, high, low)
    df['wr1'] = wr1
    df['wr2'] = wr2

    # BIAS
    bias1, bias2, bias3 = BIAS(close)
    df['bias6'] = bias1
    df['bias12'] = bias2
    df['bias24'] = bias3

    # 布林带
    upper, mid, lower = BOLL(close)
    df['boll_upper'] = upper
    df['boll_mid'] = mid
    df['boll_lower'] = lower

    # 心理线
    psy, psyma = PSY(close)
    df['psy'] = psy
    df['psyma'] = psyma

    # CCI
    cci = CCI(close, high, low)
    df['cci'] = cci

    # ATR
    atr = ATR(close, high, low)
    df['atr'] = atr

    # BBI
    bbi = BBI(close)
    df['bbi'] = bbi

    # DMI
    pdi, mdi, adx, adxr = DMI(close, high, low)
    df['dmi_pdi'] = pdi
    df['dmi_mdi'] = mdi
    df['dmi_adx'] = adx
    df['dmi_adxr'] = adxr

    # 唐安奇通道
    taq_up, taq_mid, taq_down = TAQ(high, low, 20)
    df['taq_up'] = taq_up
    df['taq_mid'] = taq_mid
    df['taq_down'] = taq_down

    # 肯特纳通道
    ktn_up, ktn_mid, ktn_down = KTN(close, high, low, 20, 10)
    df['ktn_upper'] = ktn_up
    df['ktn_mid'] = ktn_mid
    df['ktn_lower'] = ktn_down

    # TRIX
    trix, trma = TRIX(close)
    df['trix'] = trix
    df['trma'] = trma

    # VR
    vr = VR(close, volume)
    df['vr'] = vr

    # CR
    cr = CR(close, high, low)
    df['cr'] = cr

    # DPO
    dpo, madpo = DPO(close)
    df['dpo'] = dpo
    df['madpo'] = madpo

    # BRAR
    ar, br = BRAR(open_, close, high, low)
    df['brar_ar'] = ar
    df['brar_br'] = br

    # DFMA
    dfma, dfmama = DFMA(close)
    df['dfma'] = dfma
    df['dfmama'] = dfmama

    # MTM
    mtm, mtmma = MTM(close)
    df['mtm'] = mtm
    df['mtmma'] = mtmma

    # MASS
    mass, ma_mass = MASS(high, low)
    df['mass'] = mass
    df['ma_mass'] = ma_mass

    # ROC
    roc, maroc = ROC(close)
    df['roc'] = roc
    df['maroc'] = maroc

    # EXPMA
    ema12, ema50 = EXPMA(close)
    df['ema12'] = ema12
    df['ema50'] = ema50

    # OBV
    obv = OBV(close, volume)
    df['obv'] = obv

    # MFI
    try:
        mfi = MFI(close, high, low, volume)
        df['mfi'] = mfi
    except:
        # 如果出现错误，填充NaN
        df['mfi'] = np.nan

    # ASI
    asi, asit = ASI(open_, close, high, low)
    df['asi'] = asi
    df['asit'] = asit

    # 移动平均线
    df['ma5'] = MA(close, 5)
    df['ma10'] = MA(close, 10)
    df['ma20'] = MA(close, 20)
    df['ma30'] = MA(close, 30)
    df['ma60'] = MA(close, 60)

    # EMA
    df['ema12'] = EMA(close, 12)
    df['ema26'] = EMA(close, 26)

    # 其他常用指标
    df['slopec5'] = SLOPE(close, 5)  # 5日收盘价斜率
    df['slope10'] = SLOPE(close, 10)  # 10日收盘价斜率

    return df