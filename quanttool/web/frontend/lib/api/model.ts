import { api } from './index';
import type { TrainParams, ModelInfo, PredictionResult, TrainingProgress, ApiModelInfo } from '@/types/model';

interface TrainResponse {
  task_id?: string;
  success?: boolean;
  model_id?: string;
  model_path?: string;
  train_samples?: number;
  valid_samples?: number;
  test_samples?: number;
}

type BackendPredictionResult = {
  symbol?: string;
  instrument?: string;
  name?: string;
  score?: number;
  pred_score?: number;
  predicted_return?: number;
  pred_return?: number;
  confidence?: number;
  error?: string;
};

interface PredictApiResponse {
  success: boolean;
  model_path?: string;
  predictions?: BackendPredictionResult[];
}

// 将 API 返回的模型数据转换为前端格式
function transformModelData(apiModel: ApiModelInfo): ModelInfo {
  const filename = apiModel.filename || apiModel.path.split('/').pop() || apiModel.path;

  return {
    id: filename.replace('.pkl', ''),
    name: filename.replace('.pkl', ''),
    type: 'GBM',
    path: apiModel.path,
    created_at: apiModel.modified,
  };
}

function transformQrunModelData(apiModel: ApiModelInfo): ModelInfo {
  return {
    id: apiModel.path,
    name: apiModel.run_name || apiModel.run_id || apiModel.path,
    type: apiModel.model_type || 'QRun GBM',
    path: apiModel.path,
    created_at: apiModel.modified,
    params: apiModel.config,
  };
}

function normalizePredictionResult(result: BackendPredictionResult, index: number): PredictionResult {
  return {
    symbol: result.symbol || result.instrument || '',
    name: result.name || result.symbol || result.instrument || '',
    score: Number(result.score ?? result.pred_score ?? 0),
    rank: index + 1,
    predicted_return: Number(result.predicted_return ?? result.pred_return ?? 0),
    confidence: Number(result.confidence ?? 0),
  };
}

export const modelApi = {
  // 训练 GBM 模型
  train: (params: TrainParams) =>
    api.post<any, TrainResponse>('/gbm/train', params),

  // 获取训练进度
  getTrainingProgress: (taskId: string) =>
    api.get<any, TrainingProgress>(`/gbm/train/${taskId}/progress`),

  // 使用模型预测
  predict: (modelId: string, symbols?: string[]) =>
    api.post<any, PredictApiResponse>('/gbm/predict', { model_path: modelId, symbols }).then((response) => {
      return (response.predictions || []).map(normalizePredictionResult);
    }),

  // 获取模型列表
  getModels: async (): Promise<ModelInfo[]> => {
    const data = await api.get<any, ApiModelInfo[]>('/gbm/models');
    return (data || []).map(transformModelData);
  },

  // 获取 Qrun 模型列表
  getQrunModels: async (): Promise<ModelInfo[]> => {
    try {
      const data = await api.get<any, ApiModelInfo[]>('/gbm/qrun-models');
      return (data || []).map(transformQrunModelData);
    } catch {
      return [];
    }
  },

  // 删除模型
  deleteModel: (modelId: string) =>
    api.delete<any, void>(`/gbm/models/${modelId}`),
};
