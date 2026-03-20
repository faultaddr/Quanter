'use client';

import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import { cn } from '@/lib/utils';

interface TrainingProgressProps {
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  error?: string;
}

export default function TrainingProgress({
  status,
  progress,
  message,
  error,
}: TrainingProgressProps) {
  const getStatusBadge = () => {
    switch (status) {
      case 'pending':
        return <Badge variant="default">等待中</Badge>;
      case 'running':
        return <Badge variant="primary">训练中</Badge>;
      case 'completed':
        return <Badge variant="success">已完成</Badge>;
      case 'failed':
        return <Badge variant="danger">失败</Badge>;
    }
  };

  const getProgressColor = () => {
    if (status === 'failed') return 'bg-danger';
    if (status === 'completed') return 'bg-success';
    return 'bg-primary';
  };

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <span className="font-medium text-text-primary">训练进度</span>
        {getStatusBadge()}
      </div>

      {/* Progress Bar */}
      {status === 'running' && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-text-muted">{message || '正在训练...'}</span>
            <span className="text-text-secondary">{progress.toFixed(0)}%</span>
          </div>
          <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
            <div
              className={cn('h-full rounded-full transition-all duration-300', getProgressColor())}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Status Message */}
      {status === 'pending' && (
        <div className="text-center py-4 text-text-muted">
          等待训练开始...
        </div>
      )}

      {status === 'completed' && (
        <div className="text-center py-4">
          <svg className="w-12 h-12 mx-auto text-success mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="text-text-primary">训练完成</div>
        </div>
      )}

      {status === 'failed' && (
        <div className="text-center py-4">
          <svg className="w-12 h-12 mx-auto text-danger mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="text-danger">训练失败</div>
          {error && (
            <div className="text-sm text-text-muted mt-2">{error}</div>
          )}
        </div>
      )}
    </Card>
  );
}
