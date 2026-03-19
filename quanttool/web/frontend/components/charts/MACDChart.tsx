'use client';

import { useEffect, useRef, useMemo } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';

interface MACDData {
  dif: number[];
  dea: number[];
  macd: number[];
}

interface MACDChartProps {
  data: MACDData;
  dates: string[];
  height?: number;
}

export default function MACDChart({ data, dates, height = 200 }: MACDChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const { dif, dea, macd, macdColors } = useMemo(() => {
    const macdColors = data.macd.map((v) => (v >= 0 ? '#10B981' : '#EF4444'));
    return {
      dif: data.dif,
      dea: data.dea,
      macd: data.macd,
      macdColors,
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
      },
      legend: {
        data: ['DIF', 'DEA', 'MACD'],
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
          name: 'DIF',
          type: 'line',
          data: dif,
          showSymbol: false,
          lineStyle: { width: 1, color: '#F59E0B' },
        },
        {
          name: 'DEA',
          type: 'line',
          data: dea,
          showSymbol: false,
          lineStyle: { width: 1, color: '#8B5CF6' },
        },
        {
          name: 'MACD',
          type: 'bar',
          data: macd.map((v, i) => ({
            value: v,
            itemStyle: { color: macdColors[i] },
          })),
          itemStyle: {
            opacity: 0.8,
          },
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
  }, [data, dates, dif, dea, macd, macdColors]);

  return (
    <div
      ref={chartRef}
      style={{ width: '100%', height }}
    />
  );
}
