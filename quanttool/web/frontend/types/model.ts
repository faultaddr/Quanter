// 模型相关类型定义

export interface TrainParams {
  symbols: string[];
  start_date: string;
  end_date: string;
  label_days: number;
  top_k: number;
  features?: string[];
}

// 后端 API 返回的原始模型数据
export interface ApiModelInfo {
  path: string;
  filename?: string;
  run_id?: string;
  run_name?: string;
  model_type?: string;
  size_mb: number;
  modified: string;
  config?: Record<string, unknown>;
}

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  path?: string;
  created_at?: string;
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
