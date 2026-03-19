'use client';

import { useState, useEffect } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Select from '@/components/ui/Select';
import Badge from '@/components/ui/Badge';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/Table';
import { useAppStore } from '@/stores/useAppStore';
import { modelApi } from '@/lib/api/model';
import { monitorApi } from '@/lib/api/monitor';
import { useApi } from '@/hooks/useApi';
import { useToast } from '@/hooks/useToast';
import Loading from '@/components/ui/Loading';
import EmptyState from '@/components/ui/EmptyState';
import { formatNumber, formatPercent, getChangeColorClass } from '@/lib/utils';
import type { ModelInfo, PredictionResult } from '@/types/model';

export default function PicksPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);
  const toast = useToast();

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [topK, setTopK] = useState(10);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);

  const { loading: loadingModels, execute: fetchModels } = useApi(modelApi.getQrunModels);
  const { loading: loadingPredictions, execute: fetchPicks } = useApi(monitorApi.getPicks);

  useEffect(() => {
    setActivePage('picks');
    loadModels();
  }, [setActivePage]);

  const loadModels = async () => {
    const data = await fetchModels();
    if (data && data.length > 0) {
      setModels(data);
      setSelectedModel(data[0].id);
    }
  };

  const handlePredict = async () => {
    if (!selectedModel) {
      toast.warning('请选择模型');
      return;
    }

    const data = await fetchPicks(selectedModel, topK);
    if (data) {
      setPredictions(data);
      toast.success(`预测完成，推荐 ${data.length} 只股票`);
    }
  };

  const modelOptions = models.map((m) => ({
    value: m.id,
    label: m.name,
  }));

  const topKOptions = [5, 10, 20, 30].map((k) => ({
    value: String(k),
    label: `Top ${k}`,
  }));

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-text-primary">智能荐股</h1>
          <p className="text-text-muted mt-1">利用GBM机器学习模型，预测股票未来收益表现</p>
        </div>

        {/* Model Selection */}
        <Card title="选择模型">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Select
              label="选择模型"
              options={modelOptions}
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              placeholder={loadingModels ? '加载中...' : '选择模型'}
            />
            <Select
              label="推荐数量"
              options={topKOptions}
              value={String(topK)}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
            <div className="flex items-end">
              <Button
                onClick={handlePredict}
                loading={loadingPredictions}
                disabled={!selectedModel}
                className="w-full"
              >
                开始预测
              </Button>
            </div>
          </div>

          {/* Model Info */}
          {selectedModel && models.length > 0 && (
            <div className="mt-4 p-4 bg-bg-tertiary rounded-lg">
              {(() => {
                const model = models.find((m) => m.id === selectedModel);
                if (!model) return null;
                return (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-text-muted">模型类型：</span>
                      <span className="text-text-primary ml-1">{model.type}</span>
                    </div>
                    <div>
                      <span className="text-text-muted">创建时间：</span>
                      <span className="text-text-primary ml-1">{model.created_at}</span>
                    </div>
                    {model.metrics?.accuracy && (
                      <div>
                        <span className="text-text-muted">准确率：</span>
                        <span className="text-success ml-1">{(model.metrics.accuracy * 100).toFixed(1)}%</span>
                      </div>
                    )}
                    {model.metrics?.f1 && (
                      <div>
                        <span className="text-text-muted">F1分数：</span>
                        <span className="text-primary ml-1">{(model.metrics.f1 * 100).toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          )}
        </Card>

        {/* Predictions */}
        {predictions.length > 0 ? (
          <Card title={`推荐股票 (${predictions.length})`}>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>排名</TableHead>
                  <TableHead>代码</TableHead>
                  <TableHead>名称</TableHead>
                  <TableHead className="text-right">预测分数</TableHead>
                  <TableHead className="text-right">预期收益</TableHead>
                  <TableHead className="text-right">置信度</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {predictions.map((item, index) => (
                  <TableRow key={item.symbol} hoverable>
                    <TableCell>
                      {index < 3 ? (
                        <Badge variant={index === 0 ? 'primary' : index === 1 ? 'success' : 'warning'}>
                          #{index + 1}
                        </Badge>
                      ) : (
                        <span className="text-text-muted">#{index + 1}</span>
                      )}
                    </TableCell>
                    <TableCell className="font-mono">{item.symbol}</TableCell>
                    <TableCell>{item.name}</TableCell>
                    <TableCell className="text-right font-medium text-primary">
                      {formatNumber(item.score, 4)}
                    </TableCell>
                    <TableCell className={`text-right ${getChangeColorClass(item.predicted_return)}`}>
                      {item.predicted_return !== undefined ? formatPercent(item.predicted_return) : '-'}
                    </TableCell>
                    <TableCell className="text-right">
                      {item.confidence !== undefined ? (
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-16 h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${item.confidence >= 0.7 ? 'bg-success' : item.confidence >= 0.4 ? 'bg-warning' : 'bg-danger'}`}
                              style={{ width: `${item.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-text-muted w-10">
                            {(item.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      ) : '-'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        ) : (
          !loadingPredictions && models.length > 0 && (
            <EmptyState
              title="选择模型开始预测"
              description="选择一个训练好的GBM模型，获取股票推荐"
            />
          )
        )}

        {models.length === 0 && !loadingModels && (
          <EmptyState
            title="暂无可用模型"
            description="请先训练GBM模型"
            action={{
              label: '前往训练',
              onClick: () => window.location.href = '/model',
            }}
          />
        )}
      </div>
    </PageContainer>
  );
}
