'use client';

import { useEffect, useRef, useMemo } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import { CHART_COLORS } from '@/lib/constants';

interface FlowData {
  date: string;
  main_inflow: number;      // 主力流入（万元）
  main_outflow: number;     // 主力流出（万元）
  retail_inflow: number;    // 散户流入（万元）
  retail_outflow: number;   // 散户流出（万元）
  net_main: number;         // 主力净流入
  net_retail: number;       // 散户净流入
}

interface FlowChartProps {
  data: FlowData[];
  height?: number;
}

export default function FlowChart({ data, height = 400 }: FlowChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const { dates, netMain, netRetail, mainInflow, mainOutflow } = useMemo(() => {
    const dates: string[] = [];
    const netMain: number[] = [];
    const netRetail: number[] = [];
    const mainInflow: number[] = [];
    const mainOutflow: number[] = [];

    data.forEach((item) => {
      dates.push(item.date);
      netMain.push(item.net_main / 10000); // 转换为亿
      netRetail.push(item.net_retail / 10000);
      mainInflow.push(item.main_inflow / 10000);
      mainOutflow.push(-item.main_outflow / 10000); // 负数表示流出
    });

    return { dates, netMain, netRetail, mainInflow, mainOutflow };
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
        axisPointer: { type: 'cross' },
        backgroundColor: 'rgba(30, 41, 59, 0.95)',
        borderColor: '#334155',
        textStyle: { color: '#F8FAFC' },
        formatter: (params: any) => {
          const dataIndex = params[0].dataIndex;
          const item = data[dataIndex];
          return `
            <div style="padding: 4px 0;">
              <div style="font-weight: 500; margin-bottom: 4px;">${item.date}</div>
              <div style="color: #10B981;">主力净流入: ${(item.net_main / 10000).toFixed(2)}亿</div>
              <div style="color: #EF4444;">散户净流入: ${(item.net_retail / 10000).toFixed(2)}亿</div>
              <div style="margin-top: 4px;">主力流入: ${(item.main_inflow / 10000).toFixed(2)}亿</div>
              <div>主力流出: ${(item.main_outflow / 10000).toFixed(2)}亿</div>
            </div>
          `;
        },
      },
      legend: {
        data: ['主力净流入', '散户净流入'],
        textStyle: { color: '#94A3B8' },
        top: 0,
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
        axisLabel: { color: '#94A3B8' },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        name: '净流入(亿)',
        nameTextStyle: { color: '#94A3B8' },
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94A3B8' },
        splitLine: { lineStyle: { color: '#334155', type: 'dashed' } },
      },
      dataZoom: [
        {
          type: 'inside',
          start: 70,
          end: 100,
        },
        {
          show: true,
          type: 'slider',
          bottom: '2%',
          start: 70,
          end: 100,
          borderColor: '#334155',
          backgroundColor: '#1E293B',
          fillerColor: 'rgba(59, 130, 246, 0.2)',
          handleStyle: { color: '#3B82F6' },
          textStyle: { color: '#94A3B8' },
        },
      ],
      series: [
        {
          name: '主力净流入',
          type: 'bar',
          data: netMain,
          itemStyle: {
            color: (params: any) => params.value >= 0 ? '#10B981' : '#EF4444',
          },
          barWidth: '40%',
        },
        {
          name: '散户净流入',
          type: 'line',
          data: netRetail,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 2, color: '#F59E0B' },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(245, 158, 11, 0.3)' },
              { offset: 1, color: 'rgba(245, 158, 11, 0)' },
            ]),
          },
        },
      ],
    };

    chartInstance.current.setOption(option);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [data, dates, netMain, netRetail]);

  if (data.length === 0) {
    return (
      <div className="text-center py-8 text-text-muted">
        暂无资金流向数据
      </div>
    );
  }

  return <div ref={chartRef} style={{ width: '100%', height }} />;
}
