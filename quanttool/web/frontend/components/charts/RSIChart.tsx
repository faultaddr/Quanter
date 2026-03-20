'use client';

import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';

interface RSIData {
  rsi6: number[];
  rsi12: number[];
  rsi24: number[];
}

interface RSIChartProps {
  data: RSIData;
  dates: string[];
  height?: number;
}

export default function RSIChart({ data, dates, height = 200 }: RSIChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

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
      },
      legend: {
        data: ['RSI6', 'RSI12', 'RSI24'],
        top: 0,
        right: 0,
        textStyle: {
          color: '#94A3B8',
        },
      },
      grid: {
        left: '10%',
        right: '8%',
        top: '15%',
        bottom: '15%',
      },
      xAxis: {
        type: 'category',
        data: dates,
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
        min: 'dataMin',
        max: 'dataMax',
      },
      yAxis: {
        scale: true,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94A3B8' },
        splitLine: {
          lineStyle: {
            color: '#334155',
            type: 'dashed',
          },
        },
        min: 0,
        max: 100,
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          start: 50,
          end: 100,
        },
      ],
      series: [
        {
          name: 'RSI6',
          type: 'line',
          data: data.rsi6,
          showSymbol: false,
          lineStyle: { width: 1, color: '#3B82F6' },
        },
        {
          name: 'RSI12',
          type: 'line',
          data: data.rsi12,
          showSymbol: false,
          lineStyle: { width: 1, color: '#F59E0B' },
        },
        {
          name: 'RSI24',
          type: 'line',
          data: data.rsi24,
          showSymbol: false,
          lineStyle: { width: 1, color: '#8B5CF6' },
        },
      ],
      // 超买超卖区域标记
      markLines: [
        {
          silent: true,
          symbol: 'none',
          lineStyle: {
            color: '#64748B',
            type: 'dashed',
            width: 1,
          },
          label: {
            show: false,
          },
          data: [
            { yAxis: 20 },
            { yAxis: 80 },
          ],
        },
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
  }, [data, dates]);

  return (
    <div
      ref={chartRef}
      style={{ width: '100%', height }}
    />
  );
}
