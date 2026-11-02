import axios from 'axios';
import {
  Dataset,
  DatasetPreview,
  TrainConfig,
  TrainResult,
  ModelSummary,
  ModelFeatureInfo,
  PredictionResult,
  PredictionHistoryItem,
  ModelComparisonData,
  SystemHealth,
  DashboardStats,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000,
});

// Auto-retry once on transient network errors or server reload
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const config = error.config;
    if (
      (!error.response || error.code === 'ERR_NETWORK') &&
      config &&
      !config._retry
    ) {
      config._retry = true;
      await new Promise((resolve) => setTimeout(resolve, 1200));
      return apiClient(config);
    }
    return Promise.reject(error);
  }
);

// Datasets API
export const getDatasets = async (): Promise<Dataset[]> => {
  const res = await apiClient.get<Dataset[]>('/datasets');
  return res.data;
};

export const getDataset = async (id: string): Promise<Dataset> => {
  const res = await apiClient.get<Dataset>(`/datasets/${id}`);
  return res.data;
};

export const getDatasetPreview = async (id: string): Promise<DatasetPreview> => {
  const res = await apiClient.get<DatasetPreview>(`/datasets/${id}/preview`);
  return res.data;
};

export const uploadDataset = async (
  file: File,
  name?: string,
  task_type?: string,
  description?: string
): Promise<Dataset> => {
  const formData = new FormData();
  formData.append('file', file);
  if (name) formData.append('name', name);
  if (task_type) formData.append('task_type', task_type);
  if (description) formData.append('description', description);

  const res = await apiClient.post<Dataset>('/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const deleteDataset = async (id: string): Promise<{ message: string }> => {
  const res = await apiClient.delete<{ message: string }>(`/datasets/${id}`);
  return res.data;
};

// Training API
export const trainModel = async (config: TrainConfig): Promise<TrainResult> => {
  const res = await apiClient.post<TrainResult>('/models/train', config);
  return res.data;
};

export const getTrainedModels = async (): Promise<ModelSummary[]> => {
  const res = await apiClient.get<ModelSummary[]>('/models');
  return res.data;
};

export const getModelFeatureInfo = async (experimentId: string): Promise<ModelFeatureInfo> => {
  const res = await apiClient.get<ModelFeatureInfo>(`/models/${experimentId}/feature-info`);
  return res.data;
};

// Prediction API
export const predictModel = async (
  experimentId: string,
  inputs: Record<string, any>
): Promise<PredictionResult> => {
  const res = await apiClient.post<PredictionResult>(`/models/${experimentId}/predict`, {
    experiment_id: experimentId,
    inputs,
  });
  return res.data;
};

export const getModelPredictions = async (
  experimentId: string
): Promise<PredictionHistoryItem[]> => {
  const res = await apiClient.get<PredictionHistoryItem[]>(`/models/${experimentId}/predictions`);
  return res.data;
};

// Model Comparison API
export const compareModels = async (
  idA: string,
  idB: string
): Promise<ModelComparisonData> => {
  const res = await apiClient.get<ModelComparisonData>(`/models/compare/${idA}/${idB}`);
  return res.data;
};

// Experiments API
export const getExperiments = async (): Promise<ModelSummary[]> => {
  const res = await apiClient.get<ModelSummary[]>('/experiments');
  return res.data;
};

export const getExperimentDetails = async (id: string): Promise<any> => {
  const res = await apiClient.get(`/experiments/${id}`);
  return res.data;
};

export const deleteExperiment = async (id: string): Promise<{ message: string }> => {
  const res = await apiClient.delete<{ message: string }>(`/experiments/${id}`);
  return res.data;
};

// Health & System API
export const getSystemHealth = async (): Promise<SystemHealth> => {
  const res = await apiClient.get<SystemHealth>('/health');
  return res.data;
};

export const getDashboardStats = async (): Promise<DashboardStats> => {
  const res = await apiClient.get<DashboardStats>('/stats');
  return res.data;
};

// Auto Clean & Train Pipeline API
export interface CleanRequest {
  dataset_id: string;
  target_column?: string;
  handle_duplicates?: boolean;
  impute_missing?: boolean;
  cap_outliers?: boolean;
  clean_strings?: boolean;
  save_as_new_dataset?: boolean;
  new_dataset_name?: string;
}

export interface CleanResponse {
  original_dataset_id: string;
  cleaned_dataset_id?: string;
  cleaned_dataset_name?: string;
  audit_report: Record<string, any>;
  preview_rows: Record<string, any>[];
  columns: string[];
  numerical_features: string[];
  categorical_features: string[];
  recommended_target?: string;
  recommended_task: string;
}

export interface AutoTrainRequest {
  dataset_id: string;
  target_column: string;
  auto_clean_first?: boolean;
  method?: string;
  implementation_type?: string;
  test_size?: number;
  epochs?: number;
  learning_rate?: number;
}

export interface AutoTrainResponse {
  dataset_id: string;
  cleaning_report?: Record<string, any>;
  primary_model: TrainResult;
  comparison_model?: TrainResult;
  metric_differences?: Record<string, number>;
  sample_prediction?: {
    inputs: Record<string, any>;
    prediction: string;
    probabilities?: Record<string, number>;
    task_type: string;
  };
  summary_message: string;
}

export const cleanDataset = async (config: CleanRequest): Promise<CleanResponse> => {
  const res = await apiClient.post<CleanResponse>('/pipeline/clean', config);
  return res.data;
};

export const autoTrainDataset = async (config: AutoTrainRequest): Promise<AutoTrainResponse> => {
  const res = await apiClient.post<AutoTrainResponse>('/pipeline/auto-train', config);
  return res.data;
};

