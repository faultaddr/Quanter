'use client';

import { useState } from 'react';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Select from '@/components/ui/Select';
import Card from '@/components/ui/Card';
import type { TrainParams } from '@/types/model';

interface ModelFormProps {
  onSubmit: (params: TrainParams) => void;
  loading?: boolean;
}

export default function ModelForm({ onSubmit, loading }: ModelFormProps) {
  const [symbols, setSymbols] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [labelDays, setLabelDays] = useState(5);
  const [topK, setTopK] = useState(10);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const symbolList = symbols
      .split(/[,，\s]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    onSubmit({
      symbols: symbolList,
      start_date: startDate,
      end_date: endDate,
      label_days: labelDays,
      top_k: topK,
    });
  };

  return (
    <Card title="训练参数">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1.5">
            股票池
          </label>
          <textarea
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            placeholder="输入股票代码，多个用逗号分隔，如：000001, 600000"
            className="w-full bg-bg-tertiary border border-border-primary rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent min-h-[80px] resize-none"
          />
          <p className="text-xs text-text-muted mt-1">
            留空则使用沪深300成分股
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Input
            label="开始日期"
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
          <Input
            label="结束日期"
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Input
            label="预测天数"
            type="number"
            value={labelDays}
            onChange={(e) => setLabelDays(Number(e.target.value))}
            min={1}
            max={30}
          />
          <Input
            label="选股数量 (Top K)"
            type="number"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            min={1}
            max={50}
          />
        </div>

        <div className="pt-2">
          <Button type="submit" loading={loading} className="w-full">
            开始训练
          </Button>
        </div>
      </form>
    </Card>
  );
}
