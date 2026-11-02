import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Play,
  RotateCcw,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Sliders,
  TrendingDown,
  BarChart3,
  ArrowRight,
  ShieldCheck,
  Upload,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
} from 'recharts';
import { Card } from '../components/Card';
import { PlotlyChart } from '../components/PlotlyChart';
import {
  Dataset,
  DatasetPreview,
  TrainConfig,
  TrainResult,
  AlgorithmType,
  ImplementationType,
} from '../types';
import { getDatasets, getDatasetPreview, trainModel } from '../services/api';

interface TrainingLabProps {
  initialDatasetId?: string;
  onNavigateToPrediction?: (experimentId: string) => void;
  onNavigateToUpload?: () => void;
  showToast: (type: 'success' | 'error' | 'info', message: string, title?: string) => void;
}

export const TrainingLab: React.FC<TrainingLabProps> = ({
  initialDatasetId,
  onNavigateToPrediction,
  onNavigateToUpload,
  showToast,
}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(initialDatasetId || '');
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);

  // Configuration Form State
  const [algorithm, setAlgorithm] = useState<AlgorithmType>('linear_regression');
  const [implementationType, setImplementationType] = useState<ImplementationType>('scratch');
  const [experimentName, setExperimentName] = useState<string>('');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>('');

  // Hyperparameters
  const [learningRate, setLearningRate] = useState<number>(0.05);
  const [epochs, setEpochs] = useState<number>(300);
  const [testSize, setTestSize] = useState<number>(0.2);
  const [threshold, setThreshold] = useState<number>(0.5);
  const [hiddenLayersStr, setHiddenLayersStr] = useState<string>('64, 32');
  const [activation, setActivation] = useState<'relu' | 'sigmoid' | 'tanh'>('relu');
  const [optimizer, setOptimizer] = useState<'adam' | 'sgd'>('adam');
  const [batchSize, setBatchSize] = useState<number>(32);

  // Preprocessing
  const [scaling, setScaling] = useState<'standard' | 'minmax' | 'none'>('standard');
  const [imputation, setImputation] = useState<'mean' | 'median' | 'most_frequent'>('mean');
  const [encoding, setEncoding] = useState<'onehot' | 'ordinal'>('onehot');

  // Execution & Results
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);

  // 1. Fetch datasets
  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const data = await getDatasets();
        setDatasets(data);
        if (data.length > 0) {
          const defaultId = initialDatasetId || data[0].id;
          setSelectedDatasetId(defaultId);
        }
      } catch (err: any) {
        showToast('error', 'Failed to load datasets.');
      }
    };
    loadDatasets();
  }, [initialDatasetId]);

  // 2. When selected dataset changes, load its preview & features
  useEffect(() => {
    const ds = datasets.find((d) => d.id === selectedDatasetId);
    if (!ds) return;
    setSelectedDataset(ds);

    const target = ds.target_column || ds.feature_names[ds.feature_names.length - 1];
    setTargetColumn(target);

    // Default select informative features (omit target and trivial ID identifiers)
    const idRegex = /^(id|order_id|client_id|customer_id|uuid|index|unnamed: 0)$/i;
    const inputs = ds.feature_names.filter((f) => f !== target);
    const meaningfulInputs = inputs.filter((f) => !idRegex.test(f.trim()));
    setSelectedFeatures(meaningfulInputs.length > 0 ? meaningfulInputs : inputs);

    // Load preview to inspect column types and automatically select the optimal algorithm
    getDatasetPreview(ds.id)
      .then((p) => {
        setPreview(p);
        if (p.categorical_features.includes(target)) {
          setAlgorithm('mlp');
          setLearningRate(0.01);
          setEpochs(50);
        } else if (p.numerical_features.includes(target)) {
          if (ds.task_type === 'binary_classification') {
            setAlgorithm('logistic_regression');
            setLearningRate(0.1);
            setEpochs(100);
          } else {
            setAlgorithm('linear_regression');
            setLearningRate(0.05);
            setEpochs(100);
          }
        }
      })
      .catch(() => setPreview(null));
  }, [selectedDatasetId, datasets]);


  const handleTargetChange = (newTarget: string) => {
    setTargetColumn(newTarget);
    if (selectedDataset) {
      setSelectedFeatures(selectedDataset.feature_names.filter((f) => f !== newTarget));
    }
    if (preview) {
      if (preview.categorical_features.includes(newTarget)) {
        if (algorithm === 'linear_regression') {
          setAlgorithm('mlp');
          showToast(
            'info',
            `Target '${newTarget}' is text/categorical. Switched algorithm to Multilayer Perceptron (MLP).`
          );
        }
      } else if (preview.numerical_features.includes(newTarget)) {
        if (algorithm === 'logistic_regression') {
          setAlgorithm('linear_regression');
          showToast(
            'info',
            `Target '${newTarget}' is numeric continuous. Switched algorithm to Linear Regression.`
          );
        }
      }
    }
  };

  const handleFeatureToggle = (feature: string) => {
    if (selectedFeatures.includes(feature)) {
      if (selectedFeatures.length === 1) {
        showToast('info', 'At least one input feature must be selected.');
        return;
      }
      setSelectedFeatures(selectedFeatures.filter((f) => f !== feature));
    } else {
      setSelectedFeatures([...selectedFeatures, feature]);
    }
  };

  const handleSelectAllFeatures = () => {
    if (!selectedDataset) return;
    setSelectedFeatures(selectedDataset.feature_names.filter((f) => f !== targetColumn));
  };

  const handleDeselectAllFeatures = () => {
    if (!selectedDataset) return;
    const inputs = selectedDataset.feature_names.filter((f) => f !== targetColumn);
    if (inputs.length > 0) {
      setSelectedFeatures([inputs[0]]);
    }
  };

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDataset) return;
    if (selectedFeatures.length === 0) {
      showToast('error', 'Please select at least one input feature.');
      return;
    }

    // Parse hidden layers
    const parsedHidden = hiddenLayersStr
      .split(',')
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n) && n > 0);

    const config: TrainConfig = {
      dataset_id: selectedDataset.id,
      experiment_name:
        experimentName.trim() ||
        `${algorithm.toUpperCase()} (${implementationType}) - ${selectedDataset.name}`,
      algorithm,
      implementation_type: implementationType,
      input_features: selectedFeatures,
      target_column: targetColumn,
      test_size: Number(testSize),
      random_state: 42,
      scaling,
      imputation,
      encoding,
      learning_rate: Number(learningRate),
      epochs: Number(epochs),
      threshold: Number(threshold),
      hidden_layers: parsedHidden.length > 0 ? parsedHidden : [64, 32],
      activation,
      optimizer,
      batch_size: Number(batchSize),
    };

    try {
      setIsTraining(true);
      const res = await trainModel(config);
      setTrainResult(res);
      showToast(
        'success',
        `Training complete! (${res.training_time_ms} ms)`,
        `${algorithm.replace('_', ' ').toUpperCase()} Trained`
      );
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ||
        (err?.message === 'Network Error'
          ? 'Cannot connect to backend server. Make sure the API is running on port 8000.'
          : err?.message) ||
        'Training failed. Check parameters.';
      showToast('error', msg, 'Training Error');
    } finally {
      setIsTraining(false);
    }
  };

  // Prepare loss chart data
  const lossChartData = (trainResult?.loss_history || []).map((loss, idx) => ({
    epoch: idx + 1,
    train_loss: loss,
    val_loss: trainResult?.val_loss_history?.[idx],
  }));

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Configuration Controls (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          <form onSubmit={handleStartTraining}>
            <Card
              title="Experiment Setup"
              subtitle="Algorithm, parity engine, and data partitioning"
            >
              <div className="space-y-4 text-xs">
                {/* Dataset Selection */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="block font-semibold text-gray-700 dark:text-gray-300">
                      Dataset
                    </label>
                    {onNavigateToUpload && (
                      <button
                        type="button"
                        onClick={onNavigateToUpload}
                        className="text-[11px] text-brand-600 dark:text-brand-400 font-semibold hover:underline flex items-center gap-1 cursor-pointer"
                      >
                        <Upload className="w-3 h-3" />
                        <span>Upload New CSV</span>
                      </button>
                    )}
                  </div>
                  <select
                    value={selectedDatasetId}
                    onChange={(e) => setSelectedDatasetId(e.target.value)}
                    className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-2.5 text-xs text-gray-900 dark:text-gray-100 font-medium focus:ring-2 focus:ring-brand-500 focus:outline-none"
                  >
                    {datasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.task_type.replace('_', ' ')})
                      </option>
                    ))}
                  </select>
                </div>

                {/* Algorithm Choice */}
                <div>
                  <label className="block font-semibold text-gray-700 dark:text-gray-300 mb-1">
                    Algorithm
                  </label>
                  <select
                    value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value as AlgorithmType)}
                    className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-2.5 text-xs text-gray-900 dark:text-gray-100 font-medium focus:ring-2 focus:ring-brand-500 focus:outline-none"
                  >
                    <option value="linear_regression">Linear Regression (Regression)</option>
                    <option value="logistic_regression">Logistic Regression (Binary Classification)</option>
                    <option value="mlp">Multilayer Perceptron MLP (Universal)</option>
                  </select>
                </div>

                {/* Implementation Toggle: Scratch vs. Library */}
                <div>
                  <label className="block font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
                    Implementation Approach
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setImplementationType('scratch')}
                      className={`p-2.5 rounded-lg border text-left transition-all ${
                        implementationType === 'scratch'
                          ? 'border-brand-500 bg-brand-50 dark:bg-brand-950/60 text-brand-900 dark:text-brand-200 ring-2 ring-brand-500/30'
                          : 'border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 text-gray-600 dark:text-gray-400'
                      }`}
                    >
                      <div className="font-bold flex items-center justify-between">
                        <span>Scratch</span>
                        {implementationType === 'scratch' && <CheckCircle2 className="w-3.5 h-3.5 text-brand-500" />}
                      </div>
                      <p className="text-[10px] text-gray-500 dark:text-gray-400 mt-0.5">
                        Pure NumPy / PyTorch autograd
                      </p>
                    </button>

                    <button
                      type="button"
                      onClick={() => setImplementationType('library')}
                      className={`p-2.5 rounded-lg border text-left transition-all ${
                        implementationType === 'library'
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/60 text-blue-900 dark:text-blue-200 ring-2 ring-blue-500/30'
                          : 'border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 text-gray-600 dark:text-gray-400'
                      }`}
                    >
                      <div className="font-bold flex items-center justify-between">
                        <span>Library</span>
                        {implementationType === 'library' && <CheckCircle2 className="w-3.5 h-3.5 text-blue-500" />}
                      </div>
                      <p className="text-[10px] text-gray-500 dark:text-gray-400 mt-0.5">
                        Scikit-learn reference
                      </p>
                    </button>
                  </div>
                </div>

                {/* Target Column Selector */}
                {selectedDataset && (
                  <div>
                    <label className="block font-semibold text-gray-700 dark:text-gray-300 mb-1">
                      Target Variable (Y)
                    </label>
                    <select
                      value={targetColumn}
                      onChange={(e) => handleTargetChange(e.target.value)}
                      className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-2.5 text-xs text-gray-900 dark:text-gray-100 font-medium focus:ring-2 focus:ring-brand-500 focus:outline-none"
                    >
                      {selectedDataset.feature_names.map((col) => {
                        const isNum = preview?.numerical_features.includes(col);
                        const isCat = preview?.categorical_features.includes(col);
                        const tag = isNum ? ' (Numeric)' : isCat ? ' (Categorical)' : '';
                        return (
                          <option key={col} value={col}>
                            {col}{tag}
                          </option>
                        );
                      })}
                    </select>

                    {preview && preview.categorical_features.includes(targetColumn) && algorithm === 'linear_regression' && (
                      <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/60 border border-amber-300 dark:border-amber-800 text-amber-800 dark:text-amber-300 text-xs flex items-start gap-2 mt-2">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-500" />
                        <div className="space-y-1">
                          <span className="font-bold">Categorical Target:</span>
                          <p>
                            '{targetColumn}' is a text/categorical column. Linear Regression requires a numeric target.
                          </p>
                          <button
                            type="button"
                            onClick={() => setAlgorithm('mlp')}
                            className="font-bold underline text-amber-900 dark:text-amber-200 hover:text-amber-700"
                          >
                            Switch algorithm to Multilayer Perceptron (MLP)
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Input Features Selector */}
                {selectedDataset && (
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="font-semibold text-gray-700 dark:text-gray-300">
                        Input Features ({selectedFeatures.length} selected)
                      </label>
                      <div className="space-x-2 text-[10px]">
                        <button
                          type="button"
                          onClick={handleSelectAllFeatures}
                          className="text-brand-600 dark:text-brand-400 hover:underline"
                        >
                          All
                        </button>
                        <span>•</span>
                        <button
                          type="button"
                          onClick={handleDeselectAllFeatures}
                          className="text-gray-400 hover:underline"
                        >
                          Clear
                        </button>
                      </div>
                    </div>

                    <div className="max-h-36 overflow-y-auto p-2 bg-gray-50 dark:bg-gray-800/60 rounded-lg border border-gray-200 dark:border-gray-700 space-y-1">
                      {selectedDataset.feature_names
                        .filter((f) => f !== targetColumn)
                        .map((feat) => (
                          <label
                            key={feat}
                            className="flex items-center gap-2 px-1.5 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer text-xs"
                          >
                            <input
                              type="checkbox"
                              checked={selectedFeatures.includes(feat)}
                              onChange={() => handleFeatureToggle(feat)}
                              className="rounded border-gray-300 text-brand-600 focus:ring-brand-500"
                            />
                            <span className="truncate text-gray-800 dark:text-gray-200">{feat}</span>
                          </label>
                        ))}
                    </div>
                  </div>
                )}

                {/* Hyperparameters Section */}
                <div className="pt-2 border-t border-gray-100 dark:border-gray-800 space-y-3">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-gray-900 dark:text-gray-100">
                    <Sliders className="w-3.5 h-3.5 text-brand-500" />
                    <span>Hyperparameters & Scaling</span>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[11px] text-gray-500 mb-1">Learning Rate</label>
                      <input
                        type="number"
                        step="0.001"
                        min="0.0001"
                        max="1.0"
                        value={learningRate}
                        onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
                        className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-900 dark:text-gray-100 font-mono"
                      />
                    </div>

                    <div>
                      <label className="block text-[11px] text-gray-500 mb-1">Epochs</label>
                      <input
                        type="number"
                        min="10"
                        max="5000"
                        value={epochs}
                        onChange={(e) => setEpochs(parseInt(e.target.value, 10) || 100)}
                        className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-900 dark:text-gray-100 font-mono"
                      />
                    </div>
                  </div>

                  {algorithm === 'logistic_regression' && (
                    <div>
                      <div className="flex justify-between text-[11px] text-gray-500 mb-1">
                        <span>Classification Threshold</span>
                        <span className="font-mono font-bold text-brand-500">{threshold}</span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        className="w-full accent-brand-500 cursor-pointer"
                      />
                    </div>
                  )}

                  {algorithm === 'mlp' && (
                    <div className="space-y-3 pt-1">
                      <div>
                        <label className="block text-[11px] text-gray-500 mb-1">
                          Hidden Layers (comma-separated neurons)
                        </label>
                        <input
                          type="text"
                          value={hiddenLayersStr}
                          onChange={(e) => setHiddenLayersStr(e.target.value)}
                          placeholder="e.g. 64, 32"
                          className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-900 dark:text-gray-100 font-mono"
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-[11px] text-gray-500 mb-1">Activation</label>
                          <select
                            value={activation}
                            onChange={(e) => setActivation(e.target.value as any)}
                            className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-900 dark:text-gray-100"
                          >
                            <option value="relu">ReLU</option>
                            <option value="sigmoid">Sigmoid</option>
                            <option value="tanh">Tanh</option>
                          </select>
                        </div>

                        <div>
                          <label className="block text-[11px] text-gray-500 mb-1">Optimizer</label>
                          <select
                            value={optimizer}
                            onChange={(e) => setOptimizer(e.target.value as any)}
                            className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-900 dark:text-gray-100"
                          >
                            <option value="adam">Adam</option>
                            <option value="sgd">SGD (momentum 0.9)</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Preprocessing Options */}
                  <div className="grid grid-cols-3 gap-2 pt-1">
                    <div>
                      <label className="block text-[10px] text-gray-400 mb-1">Scaling</label>
                      <select
                        value={scaling}
                        onChange={(e) => setScaling(e.target.value as any)}
                        className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-1.5 text-[11px] text-gray-900 dark:text-gray-100"
                      >
                        <option value="standard">StandardScaler</option>
                        <option value="minmax">MinMaxScaler</option>
                        <option value="none">None</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-[10px] text-gray-400 mb-1">Imputation</label>
                      <select
                        value={imputation}
                        onChange={(e) => setImputation(e.target.value as any)}
                        className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-1.5 text-[11px] text-gray-900 dark:text-gray-100"
                      >
                        <option value="mean">Mean</option>
                        <option value="median">Median</option>
                        <option value="most_frequent">Mode</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-[10px] text-gray-400 mb-1">Test Split</label>
                      <select
                        value={testSize}
                        onChange={(e) => setTestSize(parseFloat(e.target.value))}
                        className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-1.5 text-[11px] text-gray-900 dark:text-gray-100"
                      >
                        <option value={0.15}>15%</option>
                        <option value={0.2}>20%</option>
                        <option value={0.25}>25%</option>
                        <option value={0.3}>30%</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Train Button */}
                <div className="pt-3">
                  <button
                    type="submit"
                    disabled={isTraining}
                    className="w-full py-3 rounded-xl bg-gradient-to-r from-brand-500 to-emerald-600 hover:from-brand-600 hover:to-emerald-700 disabled:opacity-60 text-white font-bold text-sm shadow-md shadow-brand-500/25 flex items-center justify-center gap-2 transition-all"
                  >
                    {isTraining ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        <span>Training Model...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-white" />
                        <span>Start Training Experiment</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </Card>
          </form>
        </div>

        {/* Right Column: Training Live Output & Visualizations (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {trainResult ? (
            <>
              {/* Metrics Summary Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {trainResult.task_type === 'regression' ? (
                  <>
                    <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                        R² Score
                      </span>
                      <span className="text-xl font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                        {trainResult.metrics.r2_score}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                        MAE
                      </span>
                      <span className="text-xl font-bold font-mono text-gray-900 dark:text-gray-100 mt-1 block">
                        {trainResult.metrics.mae}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                        RMSE
                      </span>
                      <span className="text-xl font-bold font-mono text-gray-900 dark:text-gray-100 mt-1 block">
                        {trainResult.metrics.rmse}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                        Training Time
                      </span>
                      <span className="text-xl font-bold font-mono text-blue-600 dark:text-blue-400 mt-1 block">
                        {trainResult.training_time_ms} ms
                      </span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                        Accuracy
                      </span>
                      <span className="text-xl font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                        {(trainResult.metrics.accuracy * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                        F1 Score
                      </span>
                      <span className="text-xl font-bold font-mono text-gray-900 dark:text-gray-100 mt-1 block">
                        {trainResult.metrics.f1_score}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                        ROC-AUC
                      </span>
                      <span className="text-xl font-bold font-mono text-purple-600 dark:text-purple-400 mt-1 block">
                        {trainResult.metrics.roc_auc ?? 'N/A'}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                        Training Time
                      </span>
                      <span className="text-xl font-bold font-mono text-blue-600 dark:text-blue-400 mt-1 block">
                        {trainResult.training_time_ms} ms
                      </span>
                    </div>
                  </>
                )}
              </div>

              {/* Loss History Chart */}
              <Card
                title="Training Loss Curve"
                subtitle={`Epoch iterations vs cost function value (${trainResult.implementation_type})`}
                action={
                  onNavigateToPrediction && (
                    <button
                      onClick={() => onNavigateToPrediction(trainResult.experiment_id)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-xs font-semibold shadow-xs transition-colors"
                    >
                      <span>Predict with Model</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  )
                }
              >
                {lossChartData.length > 0 ? (
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={lossChartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                        <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 10 }} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1F2937',
                            borderColor: '#374151',
                            borderRadius: '8px',
                            fontSize: '11px',
                            color: '#F9FAFB',
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Line
                          type="monotone"
                          dataKey="train_loss"
                          name="Train Loss"
                          stroke="#10B981"
                          strokeWidth={2}
                          dot={false}
                        />
                        {trainResult.val_loss_history?.length > 0 && (
                          <Line
                            type="monotone"
                            dataKey="val_loss"
                            name="Validation Loss"
                            stroke="#3B82F6"
                            strokeWidth={2}
                            dot={false}
                          />
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="py-12 text-center text-xs text-gray-400">
                    OLS Closed-form analytic fit (single epoch loss recorded).
                  </div>
                )}
              </Card>

              {/* Diagnostic Plot: Actual vs Predicted (Regression) OR Confusion Matrix (Classification) */}
              {trainResult.task_type === 'regression' &&
                trainResult.metrics.actual_vs_predicted && (
                  <Card
                    title="Actual vs. Predicted Evaluation"
                    subtitle="Scatter distribution on unseen test set"
                  >
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                          <XAxis dataKey="actual" name="Actual" tick={{ fontSize: 10 }} />
                          <YAxis dataKey="predicted" name="Predicted" tick={{ fontSize: 10 }} />
                          <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            contentStyle={{
                              backgroundColor: '#1F2937',
                              borderColor: '#374151',
                              borderRadius: '8px',
                              fontSize: '11px',
                              color: '#F9FAFB',
                            }}
                          />
                          <Scatter
                            name="Predictions"
                            data={trainResult.metrics.actual_vs_predicted}
                            fill="#10B981"
                          />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </Card>
                )}

              {trainResult.task_type !== 'regression' &&
                trainResult.metrics.confusion_matrix && (
                  <Card
                    title="Confusion Matrix"
                    subtitle="Classification contingency table on test split"
                  >
                    <div className="overflow-x-auto">
                      <table className="w-full text-center text-xs">
                        <thead>
                          <tr>
                            <th className="py-2 px-3 text-gray-400">Actual \ Predicted</th>
                            {trainResult.metrics.confusion_matrix.labels.map((lbl: string) => (
                              <th key={lbl} className="py-2 px-3 font-bold text-gray-700 dark:text-gray-300">
                                Class {lbl}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                          {trainResult.metrics.confusion_matrix.matrix.map(
                            (row: number[], rowIdx: number) => (
                              <tr key={rowIdx}>
                                <td className="py-2.5 px-3 font-bold text-gray-700 dark:text-gray-300">
                                  Class {trainResult.metrics.confusion_matrix.labels[rowIdx]}
                                </td>
                                {row.map((cell: number, colIdx: number) => (
                                  <td
                                    key={colIdx}
                                    className={`py-2.5 px-3 font-mono font-bold ${
                                      rowIdx === colIdx
                                        ? 'bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400'
                                        : 'text-gray-500'
                                    }`}
                                  >
                                    {cell}
                                  </td>
                                ))}
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                )}
            </>
          ) : (
            <Card className="h-full flex items-center justify-center p-12 text-center">
              <div className="max-w-md space-y-3">
                <div className="w-12 h-12 rounded-2xl bg-gray-100 dark:bg-gray-800 text-gray-400 mx-auto flex items-center justify-center">
                  <Cpu className="w-6 h-6" />
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                  Ready to Train
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Configure your algorithm parameters on the left and click "Start Training Experiment".
                  Training loss curves, test performance metrics, and evaluation plots will render here automatically.
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
