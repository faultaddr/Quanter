'use client';

import { useEffect, useRef, useMemo } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import type { KlineData } from '@/types/stock';
import { CHART_COLORS } from '@/lib/constants';

interface KlineChartProps {
  data: KlineData[];
  height?: number;
  showVolume?: boolean;
  showMA?: boolean;
}

export default function KlineChart({
  data,
  height = 400,
  showVolume = true,
  showMA = true,
}: KlineChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const { dates, ohlc, volumes, ma5, ma10, ma20, ma60 } = useMemo(() => {
    const dates: string[] = [];
    const ohlc: number[][] = [];
    const volumes: number[] = [];
    const closes: number[] = [];

    data.forEach((item) => {
      dates.push(item.date);
      ohlc.push([item.open, item.close, item.low, item.high]);
      volumes.push(item.volume);
      closes.push(item.close);
    });

    // 计算均线
    const calcMA = (period: number) => {
      const result: (number | null)[] = [];
      for (let i = 0; i < closes.length; i++) {
        if (i < period - 1) {
          result.push(null);
        } else {
          let sum = 0;
          for (let j = 0; j < period; j++) {
            sum += closes[i - j];
          }
          result.push(+(sum / period).toFixed(2));
        }
      }
      return result;
    };

    return {
      dates,
      ohlc,
      volumes,
      ma5: calcMA(5),
      ma10: calcMA(10),
      ma20: calcMA(20),
      ma60: calcMA(60),
    };
  }, [data]);

  useEffect(() => {
    if (!chartRef.current) return;

    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current, 'dark');
    }

    const option: EChartsOption = {
      backgroundColor: 'transparent',
      animation: false,
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
        backgroundColor: 'rgba(30, 41, 59, 0.95)',
        borderColor: '#334155',
        textStyle: {
          color: '#F8FAFC',
        },
        formatter: (params: any) => {
          const dataIndex = params[0].dataIndex;
          const item = data[dataIndex];
          const change = ((item.close - item.open) / item.open * 100).toFixed(2);
          const changeColor = item.close >= item.open ? '#10B981' : '#EF4444';

          return `
            <div style="padding: 4px 0;">
              <div style="font-weight: 500; margin-bottom: 4px;">${item.date}</div>
              <div>开盘: ${item.open.toFixed(2)}</div>
              <div>收盘: <span style="color: ${changeColor}">${item.close.toFixed(2)} (${change}%)</span></div>
              <div>最高: ${item.high.toFixed(2)}</div>
              <div>最低: ${item.low.toFixed(2)}</div>
              <div>成交量: ${(item.volume / 10000).toFixed(0)}万</div>
            </div>
          `;
        },
      },
      axisPointer: {
        link: [{ xAxisIndex: 'all' }],
        label: {
          backgroundColor: '#1E293B',
        },
      },
      grid: [
        {
          left: '10%',
          right: '8%',
          top: '8%',
          height: showVolume ? '55%' : '75%',
        },
        ...(showVolume ? [{
          left: '10%',
          right: '8%',
          top: '70%',
          height: '15%',
        } as const] : []),
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          boundaryGap: false,
          axisLine: { lineStyle: { color: '#334155' } },
          axisLabel: { color: '#94A3B8' },
          axisTick: { show: false },
          splitLine: { show: false },
          min: 'dataMin',
          max: 'dataMax',
        },
        ...(showVolume ? [{
          type: 'category',
          gridIndex: 1,
          data: dates,
          boundaryGap: false,
          axisLine: { show: false },
          axisLabel: { show: false },
          axisTick: { show: false },
          splitLine: { show: false },
          min: 'dataMin',
          max: 'dataMax',
        } as const] : []),
      ],
      yAxis: [
        {
          scale: true,
          axisLine: { lineStyle: { color: '#334155' } },
          axisLabel: { color: '#94A3B8' },
          splitLine: {
            lineStyle: {
              color: '#334155',
              type: 'dashed',
            },
          },
        },
        ...(showVolume ? [{
          scale: true,
          gridIndex: 1,
          splitNumber: 2,
          axisLine: { show: false },
          axisLabel: { show: false },
          splitLine: { show: false },
        } as const] : []),
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, ...(showVolume ? [1] : [])],
          start: 50,
          end: 100,
        },
        {
          show: true,
          xAxisIndex: [0, ...(showVolume ? [1] : [])],
          type: 'slider',
          bottom: '2%',
          start: 50,
          end: 100,
          borderColor: '#334155',
          backgroundColor: '#1E293B',
          fillerColor: 'rgba(59, 130, 246, 0.2)',
          handleStyle: {
            color: '#3B82F6',
          },
          textStyle: {
            color: '#94A3B8',
          },
        },
      ],
      series: [
        {
          type: 'candlestick',
          data: ohlc,
          itemStyle: {
            color: CHART_COLORS.up,
            color0: CHART_COLORS.down,
            borderColor: CHART_COLORS.up,
            borderColor0: CHART_COLORS.down,
          },
        },
        ...(showMA ? [
          {
            name: 'MA5',
            type: 'line' as const,
            data: ma5,
            smooth: true,
            showSymbol: false,
            lineStyle: { width: 1, color: CHART_COLORS.ma5 },
          },
          {
            name: 'MA10',
            type: 'line' as const,
            data: ma10,
            smooth: true,
            showSymbol: false,
            lineStyle: { width: 1, color: CHART_COLORS.ma10 },
          },
          {
            name: 'MA20',
            type: 'line' as const,
            data: ma20,
            smooth: true,
            showSymbol: false,
            lineStyle: { width: 1, color: CHART_COLORS.ma20 },
          },
          {
            name: 'MA60',
            type: 'line' as const,
            data: ma60,
            smooth: true,
            showSymbol: false,
            lineStyle: { width: 1, color: CHART_COLORS.ma60 },
          },
        ] : []),
        ...(showVolume ? [{
          name: 'Volume',
          type: 'bar' as const,
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: volumes.map((v, i) => ({
            value: v,
            itemStyle: {
              color: ohlc[i][1] >= ohlc[i][0] ? CHART_COLORS.up : CHART_COLORS.down,
            },
          })),
          itemStyle: {
            opacity: 0.8,
          },
        }] : []),
      ],
    };

    chartInstance.current.setOption(option);

    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [data, dates, ohlc, volumes, ma5, ma10, ma20, ma60, showVolume, showMA]);

  return (
    <div
      ref={chartRef}
      style={{ width: '100%', height }}
    />
  );
}
