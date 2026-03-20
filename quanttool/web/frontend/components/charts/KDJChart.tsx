'use client';

import { useEffect, useRef, useMemo } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';

interface KDJData {
  k: number[];
  d: number[];
  j: number[];
}

interface KDJChartProps {
  data: KDJData;
  dates: string[];
  height?: number;
}

export default function KDJChart({ data, dates, height = 200 }: KDJChartProps) {
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
        data: ['K', 'D', 'J'],
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
          name: 'K',
          type: 'line',
          data: data.k,
          showSymbol: false,
          lineStyle: { width: 1, color: '#3B82F6' },
        },
        {
          name: 'D',
          type: 'line',
          data: data.d,
          showSymbol: false,
          lineStyle: { width: 1, color: '#F59E0B' },
        },
        {
          name: 'J',
          type: 'line',
          data: data.j,
          showSymbol: false,
          lineStyle: { width: 1, color: '#10B981' },
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
