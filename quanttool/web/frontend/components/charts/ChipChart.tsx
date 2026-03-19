'use client';

import { useEffect, useRef, useMemo } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import type { ChipDistribution } from '@/types/stock';

interface ChipChartProps {
  data: ChipDistribution[];
  currentPrice?: number;
  height?: number;
}

export default function ChipChart({ data, currentPrice, height = 300 }: ChipChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const { prices, percents, maxPercent } = useMemo(() => {
    const prices = data.map((d) => d.price);
    const percents = data.map((d) => d.percent * 100); // 转换为百分比
    const maxPercent = Math.max(...percents);

    return { prices, percents, maxPercent };
  }, [data]);

  useEffect(() => {
    if (!chartRef.current || data.length === 0) return;

    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current, 'dark');
    }

    const option: EChartsOption = {
      backgroundColor: 'transparent',
      animation: false,
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
        backgroundColor: 'rgba(30, 41, 59, 0.95)',
        borderColor: '#334155',
        textStyle: {
          color: '#F8FAFC',
        },
        formatter: (params: any) => {
          const item = params[0];
          return `
            <div style="padding: 4px 0;">
              <div>价格: ${item.name.toFixed(2)}</div>
              <div>筹码占比: ${item.value.toFixed(2)}%</div>
            </div>
          `;
        },
      },
      grid: {
        left: '12%',
        right: '8%',
        top: '10%',
        bottom: '15%',
      },
      xAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94A3B8', formatter: '{value}%' },
        splitLine: {
          lineStyle: {
            color: '#334155',
            type: 'dashed',
          },
        },
        max: Math.ceil(maxPercent * 1.2),
      },
      yAxis: {
        type: 'category',
        data: prices.reverse(),
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: {
          color: '#94A3B8',
          formatter: (value: number) => value.toFixed(2),
        },
        splitLine: { show: false },
        inverse: true,
      },
      series: [
        {
          name: '筹码分布',
          type: 'bar',
          data: percents.reverse(),
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
              { offset: 0, color: '#3B82F6' },
              { offset: 1, color: '#60A5FA' },
            ]),
            borderRadius: [0, 4, 4, 0],
          },
          emphasis: {
            itemStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                { offset: 0, color: '#2563EB' },
                { offset: 1, color: '#3B82F6' },
              ]),
            },
          },
        },
      ],
    };

    // 添加当前价格标记
    if (currentPrice) {
      const priceIndex = prices.findIndex((p) => Math.abs(p - currentPrice) < 0.01);
      if (priceIndex !== -1) {
        (option as any).series[0].markPoint = {
          symbol: 'triangle',
          symbolSize: 10,
          itemStyle: {
            color: '#EF4444',
          },
          data: [
            {
              name: '当前价',
              coord: [percents[priceIndex], prices[priceIndex]],
            },
          ],
        };
      }
    }

    chartInstance.current.setOption(option);

    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [data, prices, percents, maxPercent, currentPrice]);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center text-text-muted" style={{ height }}>
        暂无筹码数据
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
