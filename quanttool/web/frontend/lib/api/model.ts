import api from '../api';
import type { TrainParams, ModelInfo, PredictionResult, TrainingProgress, ApiModelInfo } from '@/types/model';

// 将 API 返回的模型数据转换为前端格式
function transformModelData(apiModel: ApiModelInfo): ModelInfo {
  return {
    id: apiModel.filename.replace('.pkl', ''),
    name: apiModel.filename.replace('.pkl', ''),
    type: 'GBM',
    created_at: apiModel.modified,
  };
}

export const modelApi = {
  // 训练 GBM 模型
  train: (params: TrainParams) =>
    api.post<any, { task_id: string }>('/gbm/train', params),

  // 获取训练进度
  getTrainingProgress: (taskId: string) =>
    api.get<any, TrainingProgress>(`/gbm/train/${taskId}/progress`),

  // 使用模型预测
  predict: (modelId: string, symbols?: string[]) =>
    api.post<any, PredictionResult[]>('/gbm/predict', { model_id: modelId, symbols }),

  // 获取模型列表
  getModels: async (): Promise<ModelInfo[]> => {
    const data = await api.get<any, ApiModelInfo[]>('/gbm/models');
    return (data || []).map(transformModelData);
  },

  // 获取 Qrun 模型列表
  getQrunModels: async (): Promise<ModelInfo[]> => {
    try {
      const data = await api.get<any, ApiModelInfo[]>('/gbm/qrun-models');
      return (data || []).map(transformModelData);
    } catch {
      return [];
    }
  },

  // 删除模型
  deleteModel: (modelId: string) =>
    api.delete<any, void>(`/gbm/models/${modelId}`),
};
