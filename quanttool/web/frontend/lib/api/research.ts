import type {
  SerenityResponse,
  SerenityScorecardInput,
  SerenityScoreResult,
} from '@/types/research';
import { isAxiosError } from 'axios';
import { api } from './index';

interface ResearchApiError {
  error?: string;
  detail?: string | Array<{ msg?: string }>;
}

function getResearchErrorMessage(error: unknown): string {
  if (isAxiosError<ResearchApiError>(error)) {
    const payload = error.response?.data;
    if (payload?.error) return payload.error;

    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      const messages = detail.map((item) => item.msg).filter(Boolean);
      if (messages.length > 0) return messages.join('；');
    }
  }
  return error instanceof Error ? error.message : '研究评分请求失败';
}

export const researchApi = {
  scorecard: async (input: SerenityScorecardInput): Promise<SerenityScoreResult> => {
    try {
      const response = await api.post<any, SerenityResponse<SerenityScoreResult>>(
        '/research/serenity/scorecard',
        input
      );

      if (!response.success || !response.data) {
        throw new Error(response.error || '研究评分服务未返回有效结果');
      }

      return response.data;
    } catch (error) {
      throw new Error(getResearchErrorMessage(error));
    }
  },
};
