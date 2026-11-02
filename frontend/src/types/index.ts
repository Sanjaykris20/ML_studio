export type TaskType = 'regression' | 'binary_classification' | 'multiclass_classification';
export type AlgorithmType = 'linear_regression' | 'logistic_regression' | 'mlp';
export type ImplementationType = 'scratch' | 'library';

export interface Dataset {
  id: string;
  name: string;
  filename: string;
  task_type: TaskType;
  total_rows: number;
  total_columns: number;
  feature_names: string[];
  target_column?: string;
  is_builtin: boolean;
  description?: string;
  created_at: string;
}

export interface DatasetPreview {
  id: string;
  name: string;
  total_rows: number;
  total_columns: number;
  columns: string[];
  head_rows: Record<string, any>[];
  numerical_features: string[];
  categorical_features: string[];
  missing_values: Record<string, number>;
  duplicate_rows: number;
  summary_statistics: Record<string, Record<string, any>>;
  correlation_matrix?: Record<string, Record<string, number>>;
}

export interface TrainConfig {
  dataset_id: string;
  experiment_name?: string;
  algorithm: AlgorithmType;
  implementation_type: ImplementationType;
  input_features: string[];
  target_column: string;
  task_type?: TaskType;
  test_size: number;
  random_state: number;
  scaling: 'standard' | 'minmax' | 'none';
  imputation: 'mean' | 'median' | 'most_frequent';
  encoding: 'onehot' | 'ordinal';
  learning_rate: number;
  epochs: number;
  threshold: number;
  hidden_layers: number[];
  activation: 'relu' | 'sigmoid' | 'tanh';
  optimizer: 'adam' | 'sgd';
  batch_size: number;
}

export interface TrainResult {
  experiment_id: string;
  name: string;
  status: string;
  algorithm: AlgorithmType;
  implementation_type: ImplementationType;
  task_type: TaskType;
  metrics: Record<string, any>;
  loss_history: number[];
  val_loss_history: number[];
  training_time_ms: number;
  feature_names: string[];
  target_column: string;
  created_at: string;
}

export interface ModelSummary {
  id: string;
  name: string;
  dataset_id: string;
  dataset_name: string;
  algorithm: AlgorithmType;
  implementation_type: ImplementationType;
  task_type: TaskType;
  metrics: Record<string, any>;
  training_time_ms: number;
  created_at: string;
}

export interface ModelFeatureInfo {
  experiment_id: string;
  name: string;
  algorithm: AlgorithmType;
  implementation_type: ImplementationType;
  task_type: TaskType;
  feature_names: string[];
  target_column: string;
  feature_stats: Record<string, Record<string, any>>;
}

export interface PredictionResult {
  experiment_id: string;
  prediction: string;
  raw_value: any;
  probabilities?: Record<string, number>;
  task_type: TaskType;
  timestamp: string;
}

export interface PredictionHistoryItem {
  id: string;
  experiment_id: string;
  input_values: Record<string, any>;
  predicted_value: string;
  probabilities?: Record<string, number>;
  created_at: string;
}

export interface ModelComparisonData {
  task_type: TaskType;
  model_a: {
    id: string;
    name: string;
    algorithm: string;
    implementation_type: string;
    metrics: Record<string, any>;
    loss_history?: number[];
    val_loss_history?: number[];
    training_time_ms: number;
    created_at: string;
  };
  model_b: {
    id: string;
    name: string;
    algorithm: string;
    implementation_type: string;
    metrics: Record<string, any>;
    loss_history?: number[];
    val_loss_history?: number[];
    training_time_ms: number;
    created_at: string;
  };
  metric_differences: Record<string, number>;
  comparison_valid: boolean;
  notes?: string;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded';
  database_connected: boolean;
  database_path: string;
  pytorch_version: string;
  cuda_available: boolean;
  device_name: string;
  python_version: string;
  datasets_count: number;
  experiments_count: number;
}

export interface DashboardStats {
  total_datasets: number;
  total_experiments: number;
  total_predictions: number;
  scratch_experiments_count: number;
  library_experiments_count: number;
  recent_experiments: Array<{
    id: string;
    name: string;
    algorithm: string;
    implementation_type: string;
    task_type: string;
    training_time_ms: number;
    metrics: Record<string, any>;
    created_at: string;
  }>;
}
