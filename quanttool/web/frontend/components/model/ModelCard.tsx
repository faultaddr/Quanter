'use client';

import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';
import type { ModelInfo } from '@/types/model';
import { formatDate } from '@/lib/utils';

interface ModelCardProps {
  model: ModelInfo;
  onPredict?: (modelId: string) => void;
  onDelete?: (modelId: string) => void;
  selected?: boolean;
}

export default function ModelCard({ model, onPredict, onDelete, selected }: ModelCardProps) {
  return (
    <Card className={`transition-all ${selected ? 'border-primary ring-1 ring-primary' : ''}`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="font-medium text-text-primary">{model.name}</h4>
          <p className="text-sm text-text-muted mt-0.5">{model.id}</p>
        </div>
        <Badge variant="primary">{model.type}</Badge>
      </div>

      {/* Metrics */}
      {model.metrics && (
        <div className="grid grid-cols-2 gap-2 mb-4 text-sm">
          {model.metrics.accuracy !== undefined && (
            <div className="flex justify-between">
              <span className="text-text-muted">准确率</span>
              <span className="text-text-primary">{(model.metrics.accuracy * 100).toFixed(1)}%</span>
            </div>
          )}
          {model.metrics.precision !== undefined && (
            <div className="flex justify-between">
              <span className="text-text-muted">精确率</span>
              <span className="text-text-primary">{(model.metrics.precision * 100).toFixed(1)}%</span>
            </div>
          )}
          {model.metrics.recall !== undefined && (
            <div className="flex justify-between">
              <span className="text-text-muted">召回率</span>
              <span className="text-text-primary">{(model.metrics.recall * 100).toFixed(1)}%</span>
            </div>
          )}
          {model.metrics.f1 !== undefined && (
            <div className="flex justify-between">
              <span className="text-text-muted">F1分数</span>
              <span className="text-text-primary">{(model.metrics.f1 * 100).toFixed(1)}%</span>
            </div>
          )}
        </div>
      )}

      {/* Created Time */}
      <div className="text-xs text-text-muted mb-4">
        创建于 {formatDate(model.created_at)}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        {onPredict && (
          <Button
            size="sm"
            onClick={() => onPredict(model.id)}
            className="flex-1"
          >
            预测
          </Button>
        )}
        {onDelete && (
          <Button
            size="sm"
            variant="danger"
            onClick={() => onDelete(model.id)}
          >
            删除
          </Button>
        )}
      </div>
    </Card>
  );
}
