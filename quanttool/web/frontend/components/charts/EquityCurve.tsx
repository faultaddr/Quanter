'use client';

import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';

interface EquityPoint {
  date: string;
  value: number;
}

interface EquityCurveProps {
  data: EquityPoint[];
  benchmarkData?: EquityPoint[];
  title?: string;
  height?: number;
  showLegend?: boolean;
}

export default function EquityCurve({
  data,
  benchmarkData,
  title,
  height = 300,
  showLegend = true,
}: EquityCurveProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current || data.length === 0) return;

    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current, 'dark');
    }

    const dates = data.map((d) => d.date);
    const values = data.map((d) => d.value);
    const benchmarkValues = benchmarkData?.map((d) => d.value) || [];

    // 计算收益曲线颜色
    const getColor = (values: number[]) => {
      if (values.length < 2) return '#3B82F6';
      return values[values.length - 1] >= values[0] ? '#10B981' : '#EF4444';
    };

    const option: EChartsOption = {
      backgroundColor: 'transparent',
      animation: false,
      title: title ? {
        text: title,
        left: 'center',
        top: 0,
        textStyle: {
          color: '#F8FAFC',
          fontSize: 14,
          fontWeight: 'normal',
        },
      } : undefined,
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
          const date = params[0].axisValue;
          let html = `<div style="font-weight: 500; margin-bottom: 4px;">${date}</div>`;

          params.forEach((param: any) => {
            const color = param.color;
            const name = param.seriesName;
            const value = param.value;
            const returnPct = ((value / (param.seriesIndex === 0 ? data[0].value : benchmarkData?.[0]?.value || 1) - 1) * 100).toFixed(2);
            html += `<div style="color: ${color};">${name}: ${value.toLocaleString()} (${returnPct}%)</div>`;
          });

          return html;
        },
      },
      legend: showLegend ? {
        data: benchmarkData ? ['策略', '基准'] : ['策略'],
        top: title ? 30 : 0,
        right: 0,
        textStyle: {
          color: '#94A3B8',
        },
      } : undefined,
      grid: {
        left: '10%',
        right: '8%',
        top: title ? (showLegend ? '18%' : '12%') : (showLegend ? '12%' : '8%'),
        bottom: '12%',
      },
      xAxis: {
        type: 'category',
        data: dates,
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: {
          color: '#94A3B8',
          rotate: 45,
        },
        axisTick: { show: false },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: {
          color: '#94A3B8',
          formatter: (value: number) => {
            if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
            if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
            return value.toFixed(0);
          },
        },
        splitLine: {
          lineStyle: {
            color: '#334155',
            type: 'dashed',
          },
        },
      },
      series: [
        {
          name: '策略',
          type: 'line',
          data: values,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            width: 2,
            color: getColor(values),
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: getColor(values) === '#10B981' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)' },
              { offset: 1, color: 'rgba(0, 0, 0, 0)' },
            ]),
          },
        },
        ...(benchmarkData ? [{
          name: '基准',
          type: 'line' as const,
          data: benchmarkValues,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            width: 1,
            color: '#64748B',
            type: 'dashed' as const,
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
  }, [data, benchmarkData, title, showLegend]);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center text-text-muted" style={{ height }}>
        暂无数据
      </div>
    );
  }

  return (
    <div
      ref={chartRef}
      style={{ width: '100%', height }}
    />
  );
}
