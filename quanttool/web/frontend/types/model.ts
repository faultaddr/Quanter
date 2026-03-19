// 模型相关类型定义

export interface TrainParams {
  symbols: string[];
  start_date: string;
  end_date: string;
  label_days: number;
  top_k: number;
  features?: string[];
}

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  created_at: string;
  metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
  };
  params?: Record<string, unknown>;
}

export interface TrainingProgress {
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  error?: string;
  model_id?: string;
}

export interface PredictionResult {
  symbol: string;
  name: string;
  score: number;
  rank?: number;
  predicted_return?: number;
  confidence?: number;
}
