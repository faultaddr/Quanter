import type {
  SerenityResponse,
  SerenityScorecardInput,
  SerenityScoreResult,
} from '@/types/research';
import { api } from './index';

export const researchApi = {
  scorecard: async (input: SerenityScorecardInput): Promise<SerenityScoreResult> => {
    const response = await api.post<any, SerenityResponse<SerenityScoreResult>>(
      '/research/serenity/scorecard',
      input
    );

    if (!response.success || !response.data) {
      throw new Error(response.error || '研究评分服务未返回有效结果');
    }

    return response.data;
  },
};
