'use client';

import { useState, useEffect, useCallback } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import ModelCard from '@/components/model/ModelCard';
import ModelForm from '@/components/model/ModelForm';
import TrainingProgress from '@/components/model/TrainingProgress';
import { useAppStore } from '@/stores/useAppStore';
import { modelApi } from '@/lib/api/model';
import { useApi } from '@/hooks/useApi';
import { useToast } from '@/hooks/useToast';
import Loading from '@/components/ui/Loading';
import EmptyState from '@/components/ui/EmptyState';
import type { ModelInfo, TrainParams, TrainingProgress as TrainingProgressType } from '@/types/model';

export default function ModelPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);
  const toast = useToast();

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [qrunModels, setQrunModels] = useState<ModelInfo[]>([]);
  const [activeTab, setActiveTab] = useState<'models' | 'train'>('models');
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgressType | null>(null);
  const [training, setTraining] = useState(false);

  const { loading: loadingModels, execute: fetchModels } = useApi(modelApi.getModels);
  const { loading: loadingQrunModels, execute: fetchQrunModels } = useApi(modelApi.getQrunModels);

  const loadModels = useCallback(async () => {
    const [modelsData, qrunModelsData] = await Promise.all([
      fetchModels(),
      fetchQrunModels(),
    ]);
    if (modelsData) setModels(modelsData);
    if (qrunModelsData) setQrunModels(qrunModelsData);
  }, [fetchModels, fetchQrunModels]);

  useEffect(() => {
    setActivePage('model');
    loadModels();
  }, [setActivePage, loadModels]);

  const handleTrain = async (params: TrainParams) => {
    setTraining(true);
    setTrainingProgress({ status: 'pending', progress: 0 });

    try {
      const result = await modelApi.train(params);
      if (result.task_id) {
        // Poll for progress
        const pollProgress = async () => {
          try {
            const progress = await modelApi.getTrainingProgress(result.task_id);
            setTrainingProgress(progress);

            if (progress.status === 'running') {
              setTimeout(pollProgress, 2000);
            } else if (progress.status === 'completed') {
              toast.success('模型训练完成');
              loadModels();
              setTraining(false);
            } else if (progress.status === 'failed') {
              toast.error(progress.error || '训练失败');
              setTraining(false);
            }
          } catch (error) {
            console.error('Failed to get training progress:', error);
            setTraining(false);
          }
        };

        pollProgress();
      }
    } catch (error) {
      toast.error('启动训练失败');
      setTraining(false);
    }
  };

  const handlePredict = async (modelId: string) => {
    toast.info('正在预测...');
    try {
      const results = await modelApi.predict(modelId);
      if (results) {
        toast.success(`预测完成，共 ${results.length} 只股票`);
      }
    } catch (error) {
      toast.error('预测失败');
    }
  };

  const handleDelete = async (modelId: string) => {
    if (!confirm('确定要删除此模型吗？')) return;

    try {
      await modelApi.deleteModel(modelId);
      toast.success('模型已删除');
      loadModels();
    } catch (error) {
      toast.error('删除失败');
    }
  };

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-text-primary">ML模型</h1>
          <p className="text-text-muted mt-1">训练GBM模型，管理模型生命周期</p>
        </div>

        {/* Tabs */}
        <div className="flex gap-2">
          <Button
            variant={activeTab === 'models' ? 'primary' : 'secondary'}
            onClick={() => setActiveTab('models')}
          >
            模型管理
          </Button>
          <Button
            variant={activeTab === 'train' ? 'primary' : 'secondary'}
            onClick={() => setActiveTab('train')}
          >
            训练新模型
          </Button>
        </div>

        {activeTab === 'train' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ModelForm onSubmit={handleTrain} loading={training} />

            {trainingProgress && (
              <TrainingProgress
                status={trainingProgress.status}
                progress={trainingProgress.progress}
                message={trainingProgress.message}
                error={trainingProgress.error}
              />
            )}
          </div>
        )}

        {activeTab === 'models' && (
          <>
            {/* Qrun Models */}
            {qrunModels.length > 0 && (
              <div>
                <h3 className="text-lg font-medium text-text-primary mb-3">Qrun 模型</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {qrunModels.map((model) => (
                    <ModelCard
                      key={model.id}
                      model={model}
                      onPredict={handlePredict}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Custom Models */}
            <div>
              <h3 className="text-lg font-medium text-text-primary mb-3">自定义模型</h3>
              {loadingModels ? (
                <div className="flex justify-center py-10">
                  <Loading />
                </div>
              ) : models.length === 0 ? (
                <EmptyState
                  title="暂无模型"
                  description="训练您的第一个GBM模型"
                  action={{
                    label: '训练模型',
                    onClick: () => setActiveTab('train'),
                  }}
                />
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {models.map((model) => (
                    <ModelCard
                      key={model.id}
                      model={model}
                      onPredict={handlePredict}
                      onDelete={handleDelete}
                    />
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </PageContainer>
  );
}
