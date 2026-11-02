import React, { useState, useEffect } from 'react';
import {
  Wand2,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Database,
  ArrowRight,
  ShieldCheck,
  Zap,
  Play,
  RotateCcw,
  Sliders,
  TrendingDown,
  Layers,
  FileCheck,
  Upload,
  FileUp,
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
} from 'recharts';
import { Card } from '../components/Card';
import { Dataset } from '../types';
import {
  getDatasets,
  cleanDataset,
  autoTrainDataset,
  uploadDataset,
  CleanResponse,
  AutoTrainResponse,
  predictModel,
} from '../services/api';


interface AutoCleanTrainProps {
  showToast: (type: 'success' | 'error' | 'info', message: string, title?: string) => void;
  onNavigateToPrediction?: (experimentId: string) => void;
}

export const AutoCleanTrain: React.FC<AutoCleanTrainProps> = ({
  showToast,
  onNavigateToPrediction,
}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [targetColumn, setTargetColumn] = useState<string>('');

  // Cleaning options
  const [handleDuplicates, setHandleDuplicates] = useState<boolean>(true);
  const [imputeMissing, setImputeMissing] = useState<boolean>(true);
  const [capOutliers, setCapOutliers] = useState<boolean>(true);
  const [cleanStrings, setCleanStrings] = useState<boolean>(true);

  // Cleaning state
  const [isCleaning, setIsCleaning] = useState<boolean>(false);
  const [cleanResult, setCleanResult] = useState<CleanResponse | null>(null);

  // Method Selection
  const [chosenMethod, setChosenMethod] = useState<string>('auto_recommend');
  const [implementationType, setImplementationType] = useState<string>('scratch');
  const [epochs, setEpochs] = useState<number>(300);
  const [testSplit, setTestSplit] = useState<number>(0.2);

  // Training & Output state
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainOutput, setTrainOutput] = useState<AutoTrainResponse | null>(null);

  // Live Instant Predictor state
  const [predictorInputs, setPredictorInputs] = useState<Record<string, any>>({});
  const [livePrediction, setLivePrediction] = useState<any>(null);
  const [isLivePredicting, setIsLivePredicting] = useState<boolean>(false);

  // Direct CSV Upload State
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [intakeMode, setIntakeMode] = useState<'select' | 'upload'>('select');

  const handleDirectUpload = async (file: File) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showToast('error', 'Only CSV files are supported.');
      return;
    }
    try {
      setIsUploading(true);
      const newDs = await uploadDataset(file);
      showToast('success', `Dataset '${newDs.name}' uploaded successfully!`);
      const refreshed = await getDatasets();
      setDatasets(refreshed);
      setSelectedDatasetId(newDs.id);
      setIntakeMode('select');
    } catch (err: any) {
      let msg = err?.response?.data?.detail;
      if (!msg) {
        if (err?.code === 'ERR_NETWORK' || err?.message === 'Network Error') {
          msg = 'Cannot reach backend server. Please verify the Python API server is running on port 8000.';
        } else if (err?.response?.status === 500) {
          msg = 'Backend server temporarily unavailable or encountered an error. Please retry.';
        } else {
          msg = err?.message || 'Failed to upload CSV.';
        }
      }
      showToast('error', typeof msg === 'string' ? msg : JSON.stringify(msg), 'Upload Error');
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const data = await getDatasets();
        setDatasets(data);
        if (data.length > 0) {
          setSelectedDatasetId(data[0].id);
        }
      } catch (err) {
        showToast('error', 'Failed to fetch datasets.');
      }
    };
    fetchDatasets();
  }, []);

  const activeDataset = datasets.find((d) => d.id === selectedDatasetId);

  useEffect(() => {
    if (activeDataset) {
      setTargetColumn(
        activeDataset.target_column ||
          activeDataset.feature_names[activeDataset.feature_names.length - 1]
      );
      setCleanResult(null);
      setTrainOutput(null);
      setLivePrediction(null);
    }
  }, [selectedDatasetId, activeDataset]);

  // Handle Clean Dataset action
  const handleCleanDataset = async () => {
    if (!selectedDatasetId) return;
    try {
      setIsCleaning(true);
      const res = await cleanDataset({
        dataset_id: selectedDatasetId,
        target_column: targetColumn,
        handle_duplicates: handleDuplicates,
        impute_missing: imputeMissing,
        cap_outliers: capOutliers,
        clean_strings: cleanStrings,
        save_as_new_dataset: true,
      });
      setCleanResult(res);
      showToast('success', 'Dataset cleaned and profiled successfully!');
      // Refresh datasets list to include newly cleaned dataset
      const refreshed = await getDatasets();
      setDatasets(refreshed);
      if (res.cleaned_dataset_id) {
        setSelectedDatasetId(res.cleaned_dataset_id);
      }
    } catch (err: any) {
      showToast('error', err?.response?.data?.detail || 'Cleaning failed.');
    } finally {
      setIsCleaning(false);
    }
  };

  // Handle Auto Train & Direct Output
  const handleAutoTrain = async () => {
    if (!selectedDatasetId) return;
    try {
      setIsTraining(true);
      const res = await autoTrainDataset({
        dataset_id: selectedDatasetId,
        target_column: targetColumn,
        auto_clean_first: cleanResult === null, // Clean first if user hasn't manually clicked clean
        method: chosenMethod,
        implementation_type: implementationType,
        epochs,
        test_size: testSplit,
      });
      setTrainOutput(res);
      if (res.cleaning_report && !cleanResult) {
        setCleanResult({
          original_dataset_id: selectedDatasetId,
          audit_report: res.cleaning_report,
          preview_rows: [],
          columns: [],
          numerical_features: [],
          categorical_features: [],
          recommended_task: res.primary_model.task_type,
        });
      }

      // Initialize Live Predictor with sample values
      if (res.sample_prediction) {
        setPredictorInputs(res.sample_prediction.inputs || {});
        setLivePrediction({
          prediction: res.sample_prediction.prediction,
          probabilities: res.sample_prediction.probabilities,
        });
      }

      showToast('success', res.summary_message, 'Training Complete');
    } catch (err: any) {
      showToast('error', err?.response?.data?.detail || 'Auto-train failed.');
    } finally {
      setIsTraining(false);
    }
  };

  // Live instant predictor input change
  const handleLiveInputChange = (feat: string, val: any) => {
    setPredictorInputs((prev) => ({ ...prev, [feat]: val }));
  };

  const handleRunLivePrediction = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!trainOutput?.primary_model?.experiment_id) return;
    try {
      setIsLivePredicting(true);
      const res = await predictModel(
        trainOutput.primary_model.experiment_id,
        predictorInputs
      );
      setLivePrediction(res);
      showToast('success', `Predicted: ${res.prediction}`);
    } catch (err: any) {
      showToast('error', err?.response?.data?.detail || 'Prediction failed.');
    } finally {
      setIsLivePredicting(false);
    }
  };

  // Prepare loss curves
  const lossData = (trainOutput?.primary_model?.loss_history || []).map(
    (loss, idx) => ({
      epoch: idx + 1,
      train_loss: loss,
      val_loss: trainOutput.primary_model.val_loss_history?.[idx],
    })
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-emerald-950 via-gray-900 to-gray-900 border border-emerald-800/40 text-white shadow-md">
        <div className="flex items-center gap-2.5 text-emerald-400 text-xs font-bold uppercase tracking-wider mb-2">
          <Wand2 className="w-4 h-4" />
          <span>Automated Data Cleaning & Model Training Pipeline</span>
        </div>
        <h2 className="text-2xl font-extrabold tracking-tight">
          Intake Dataset → Automated Cleaning → Choose Method → Instant Output
        </h2>
        <p className="mt-1 text-xs text-gray-300 max-w-2xl">
          Provide any dataset. The engine automatically handles missing values, removes duplicates, converts dirty strings,
          caps extreme outliers, and lets you select the optimal training method to get immediate model outputs and test predictions.
        </p>
      </div>

      {/* Step 1: Intake & Cleaning Controls */}
      <Card
        title="Step 1: Dataset Intake & Automated Cleaning"
        subtitle="Select dataset, target column, and configure automated data cleaning rules"
        action={
          <button
            onClick={handleCleanDataset}
            disabled={isCleaning}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 text-white text-xs font-bold transition-all shadow-sm"
          >
            {isCleaning ? (
              <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <Sparkles className="w-3.5 h-3.5" />
            )}
            <span>{isCleaning ? 'Cleaning Data...' : 'Auto-Clean Dataset'}</span>
          </button>
        }
      >
        <div className="space-y-4">
          {/* Intake Mode Switcher */}
          <div className="flex items-center gap-2 p-1 rounded-lg bg-gray-100 dark:bg-gray-800 w-fit text-xs font-semibold">
            <button
              type="button"
              onClick={() => setIntakeMode('select')}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                intakeMode === 'select'
                  ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-xs'
                  : 'text-gray-500 hover:text-gray-800 dark:hover:text-gray-200'
              }`}
            >
              <Database className="w-3.5 h-3.5 text-brand-500" />
              <span>Select Existing Dataset ({datasets.length})</span>
            </button>
            <button
              type="button"
              onClick={() => setIntakeMode('upload')}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                intakeMode === 'upload'
                  ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-xs'
                  : 'text-gray-500 hover:text-gray-800 dark:hover:text-gray-200'
              }`}
            >
              <Upload className="w-3.5 h-3.5 text-emerald-500" />
              <span>Upload New CSV Dataset</span>
            </button>
          </div>

          {/* Upload Dropzone View */}
          {intakeMode === 'upload' ? (
            <div className="p-6 border-2 border-dashed border-emerald-400/60 dark:border-emerald-600/40 rounded-xl bg-emerald-50/40 dark:bg-emerald-950/20 text-center space-y-3">
              <input
                type="file"
                ref={fileInputRef}
                accept=".csv"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleDirectUpload(file);
                }}
                className="hidden"
              />
              <div className="w-12 h-12 rounded-full bg-emerald-100 dark:bg-emerald-900/60 text-emerald-600 dark:text-emerald-300 mx-auto flex items-center justify-center">
                <FileUp className="w-6 h-6" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100">
                  Select or Drop a CSV Dataset File Here
                </h4>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Supports comma-delimited tabular CSV files up to 25MB.
                </p>
              </div>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 text-white text-xs font-bold transition-all shadow-sm"
              >
                {isUploading ? (
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <Upload className="w-3.5 h-3.5" />
                )}
                <span>{isUploading ? 'Uploading & Ingesting...' : 'Browse CSV from Computer'}</span>
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Dataset Dropdown */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300">
                    Select Dataset
                  </label>
                  <button
                    type="button"
                    onClick={() => setIntakeMode('upload')}
                    className="text-[11px] text-brand-600 dark:text-brand-400 font-semibold hover:underline flex items-center gap-1"
                  >
                    <Upload className="w-3 h-3" />
                    <span>Upload New CSV</span>
                  </button>
                </div>
                <select
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                  className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-2.5 text-xs text-gray-900 dark:text-gray-100 font-medium focus:ring-2 focus:ring-brand-500 focus:outline-none"
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.total_rows} rows • {d.task_type.replace('_', ' ')})
                    </option>
                  ))}
                </select>
              </div>

              {/* Target Column Dropdown */}
              <div>
                <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
                  Target Column (Output Variable to Predict)
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-2.5 text-xs text-gray-900 dark:text-gray-100 font-medium focus:ring-2 focus:ring-brand-500 focus:outline-none"
                >
                  {activeDataset?.feature_names.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}


          {/* Cleaning Rules Checkboxes */}
          <div className="pt-2 border-t border-gray-100 dark:border-gray-800">
            <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block mb-2">
              Automated Cleaning Rules
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <label className="flex items-center gap-2 p-2.5 rounded-lg bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={handleDuplicates}
                  onChange={(e) => setHandleDuplicates(e.target.checked)}
                  className="rounded text-brand-600 focus:ring-brand-500"
                />
                <span className="text-gray-800 dark:text-gray-200">Remove Duplicates</span>
              </label>

              <label className="flex items-center gap-2 p-2.5 rounded-lg bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={imputeMissing}
                  onChange={(e) => setImputeMissing(e.target.checked)}
                  className="rounded text-brand-600 focus:ring-brand-500"
                />
                <span className="text-gray-800 dark:text-gray-200">Smart Imputation</span>
              </label>

              <label className="flex items-center gap-2 p-2.5 rounded-lg bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={cleanStrings}
                  onChange={(e) => setCleanStrings(e.target.checked)}
                  className="rounded text-brand-600 focus:ring-brand-500"
                />
                <span className="text-gray-800 dark:text-gray-200">Sanitize Strings & Currency</span>
              </label>

              <label className="flex items-center gap-2 p-2.5 rounded-lg bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={capOutliers}
                  onChange={(e) => setCapOutliers(e.target.checked)}
                  className="rounded text-brand-600 focus:ring-brand-500"
                />
                <span className="text-gray-800 dark:text-gray-200">Cap IQR Outliers</span>
              </label>
            </div>
          </div>

          {/* Cleaning Audit Report (if cleaned) */}
          {cleanResult && (
            <div className="mt-3 p-4 rounded-xl bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800/40 text-xs space-y-2 animate-in fade-in">
              <div className="flex items-center gap-2 font-bold text-emerald-800 dark:text-emerald-300">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                <span>Data Cleaning Audit Summary</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                <div>
                  <span className="text-gray-500">Initial Rows:</span>{' '}
                  <span className="font-bold">{cleanResult.audit_report.initial_rows}</span>
                </div>
                <div>
                  <span className="text-gray-500">Duplicates Removed:</span>{' '}
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">
                    {cleanResult.audit_report.dropped_duplicates}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Missing Imputed:</span>{' '}
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">
                    {cleanResult.audit_report.imputed_cells}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Outliers Capped:</span>{' '}
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">
                    {cleanResult.audit_report.outliers_capped}
                  </span>
                </div>
              </div>
              {cleanResult.audit_report.actions_log?.length > 0 && (
                <div className="pt-1.5 border-t border-emerald-200/60 dark:border-emerald-800/30 space-y-1 text-[11px] text-gray-700 dark:text-gray-300">
                  {cleanResult.audit_report.actions_log.map((log: string, idx: number) => (
                    <div key={idx} className="flex items-start gap-1.5">
                      <span className="text-emerald-500">•</span>
                      <span>{log}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </Card>

      {/* Step 2: Choose Training Method */}
      <Card
        title="Step 2: Choose Training Method"
        subtitle="Select your model training architecture and execution approach"
      >
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {/* Option 1: Auto-Recommended Best Method */}
            <div
              onClick={() => setChosenMethod('auto_recommend')}
              className={`p-4 rounded-xl border cursor-pointer transition-all ${
                chosenMethod === 'auto_recommend'
                  ? 'border-brand-500 bg-brand-50/70 dark:bg-brand-950/50 ring-2 ring-brand-500/30 shadow-xs'
                  : 'border-gray-200 dark:border-gray-800 hover:border-gray-300 dark:hover:border-gray-700'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-brand-500" />
                  Auto-Recommend
                </span>
                <span className="px-1.5 py-0.5 rounded text-[9px] bg-brand-500 text-white font-bold uppercase">
                  Best
                </span>
              </div>
              <p className="text-[11px] text-gray-500 dark:text-gray-400">
                Automatically detects task type (Regression vs. Classification) and selects optimal algorithm.
              </p>
            </div>

            {/* Option 2: From Scratch Implementation */}
            <div
              onClick={() => {
                setChosenMethod('scratch_linear');
                setImplementationType('scratch');
              }}
              className={`p-4 rounded-xl border cursor-pointer transition-all ${
                chosenMethod.includes('scratch')
                  ? 'border-purple-500 bg-purple-50/70 dark:bg-purple-950/40 ring-2 ring-purple-500/30 shadow-xs'
                  : 'border-gray-200 dark:border-gray-800 hover:border-gray-300 dark:hover:border-gray-700'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-purple-500" />
                  Pure Scratch
                </span>
                <span className="px-1.5 py-0.5 rounded text-[9px] bg-purple-100 dark:bg-purple-900/60 text-purple-700 dark:text-purple-300 font-bold uppercase">
                  NumPy/PyTorch
                </span>
              </div>
              <p className="text-[11px] text-gray-500 dark:text-gray-400">
                Mathematical gradient descent and autograd backpropagation built from scratch.
              </p>
            </div>

            {/* Option 3: Standard Library Reference */}
            <div
              onClick={() => {
                setChosenMethod('library_linear');
                setImplementationType('library');
              }}
              className={`p-4 rounded-xl border cursor-pointer transition-all ${
                chosenMethod.includes('library')
                  ? 'border-blue-500 bg-blue-50/70 dark:bg-blue-950/40 ring-2 ring-blue-500/30 shadow-xs'
                  : 'border-gray-200 dark:border-gray-800 hover:border-gray-300 dark:hover:border-gray-700'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-blue-500" />
                  Library Model
                </span>
                <span className="px-1.5 py-0.5 rounded text-[9px] bg-blue-100 dark:bg-blue-900/60 text-blue-700 dark:text-blue-300 font-bold uppercase">
                  Scikit-learn
                </span>
              </div>
              <p className="text-[11px] text-gray-500 dark:text-gray-400">
                Production-standard Scikit-learn reference implementations.
              </p>
            </div>

            {/* Option 4: Dual Benchmark (Compare Both) */}
            <div
              onClick={() => setChosenMethod('compare_both')}
              className={`p-4 rounded-xl border cursor-pointer transition-all ${
                chosenMethod === 'compare_both'
                  ? 'border-emerald-500 bg-emerald-50/70 dark:bg-emerald-950/40 ring-2 ring-emerald-500/30 shadow-xs'
                  : 'border-gray-200 dark:border-gray-800 hover:border-gray-300 dark:hover:border-gray-700'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-emerald-500" />
                  Compare Both
                </span>
                <span className="px-1.5 py-0.5 rounded text-[9px] bg-emerald-100 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300 font-bold uppercase">
                  Dual Parity
                </span>
              </div>
              <p className="text-[11px] text-gray-500 dark:text-gray-400">
                Trains BOTH Scratch and Library models simultaneously on the same test split to show metric deltas.
              </p>
            </div>
          </div>

          {/* Quick Hyperparameters */}
          <div className="grid grid-cols-2 gap-4 text-xs pt-1">
            <div>
              <label className="block text-gray-500 mb-1">Epochs (Training Iterations)</label>
              <input
                type="number"
                min="10"
                max="2000"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value, 10) || 100)}
                className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-2 text-xs font-mono"
              />
            </div>

            <div>
              <label className="block text-gray-500 mb-1">Test Split Ratio</label>
              <select
                value={testSplit}
                onChange={(e) => setTestSplit(parseFloat(e.target.value))}
                className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-2 text-xs font-mono"
              >
                <option value={0.15}>15% Test Split</option>
                <option value={0.20}>20% Test Split</option>
                <option value={0.25}>25% Test Split</option>
                <option value={0.30}>30% Test Split</option>
              </select>
            </div>
          </div>

          {/* Execution Button */}
          <div className="pt-2">
            <button
              onClick={handleAutoTrain}
              disabled={isTraining}
              className="w-full py-3.5 rounded-xl bg-gradient-to-r from-brand-500 via-emerald-600 to-emerald-700 hover:from-brand-600 hover:to-emerald-800 disabled:opacity-60 text-white font-extrabold text-sm shadow-md shadow-brand-500/25 flex items-center justify-center gap-2 transition-all"
            >
              {isTraining ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Cleaning Data & Training Selected Model...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Train & Generate Model Outputs</span>
                </>
              )}
            </button>
          </div>
        </div>
      </Card>

      {/* Step 3: Model Output & Live Prediction */}
      {trainOutput && (
        <div className="space-y-6 animate-in fade-in duration-300">
          <Card
            title="Step 3: Training Output & Evaluation Results"
            subtitle={trainOutput.summary_message}
          >
            <div className="space-y-6">
              {/* Primary Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {trainOutput.primary_model.task_type === 'regression' ? (
                  <>
                    <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">
                        R² Score
                      </span>
                      <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                        {trainOutput.primary_model.metrics.r2_score}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">
                        RMSE
                      </span>
                      <span className="text-2xl font-bold font-mono text-gray-900 dark:text-gray-100 mt-1 block">
                        {trainOutput.primary_model.metrics.rmse}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">
                        MAE
                      </span>
                      <span className="text-2xl font-bold font-mono text-gray-900 dark:text-gray-100 mt-1 block">
                        {trainOutput.primary_model.metrics.mae}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">
                        Training Time
                      </span>
                      <span className="text-2xl font-bold font-mono text-blue-600 dark:text-blue-400 mt-1 block">
                        {trainOutput.primary_model.training_time_ms} ms
                      </span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">
                        Accuracy
                      </span>
                      <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                        {(trainOutput.primary_model.metrics.accuracy * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">
                        F1 Score
                      </span>
                      <span className="text-2xl font-bold font-mono text-gray-900 dark:text-gray-100 mt-1 block">
                        {trainOutput.primary_model.metrics.f1_score}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">
                        ROC-AUC
                      </span>
                      <span className="text-2xl font-bold font-mono text-purple-600 dark:text-purple-400 mt-1 block">
                        {trainOutput.primary_model.metrics.roc_auc ?? 'N/A'}
                      </span>
                    </div>
                    <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700">
                      <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">
                        Training Time
                      </span>
                      <span className="text-2xl font-bold font-mono text-blue-600 dark:text-blue-400 mt-1 block">
                        {trainOutput.primary_model.training_time_ms} ms
                      </span>
                    </div>
                  </>
                )}
              </div>

              {/* Loss Curve */}
              {lossData.length > 0 && (
                <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/40 border border-gray-200 dark:border-gray-700">
                  <span className="text-xs font-bold text-gray-800 dark:text-gray-200 block mb-2">
                    Training Loss Dynamics
                  </span>
                  <div className="h-56 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={lossData}>
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
                        {trainOutput.primary_model.val_loss_history?.length > 0 && (
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
                </div>
              )}

              {/* Dual Comparison Delta Table (if compare_both chosen) */}
              {trainOutput.comparison_model && trainOutput.metric_differences && (
                <div className="p-4 rounded-xl bg-purple-50/50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-800/40 space-y-2">
                  <span className="text-xs font-bold text-purple-900 dark:text-purple-200 uppercase tracking-wider block">
                    Dual Parity Comparison (Scratch vs. Library)
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-xs">
                    {Object.entries(trainOutput.metric_differences).map(([k, delta]) => (
                      <div key={k} className="p-2 rounded bg-white dark:bg-gray-800 border">
                        <span className="text-gray-400 capitalize block text-[10px]">{k.replace('_', ' ')}</span>
                        <span className="font-bold text-gray-900 dark:text-gray-100">
                          {delta > 0 ? `+${delta}` : delta}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Live Instant Predictor on the Trained Model */}
          <Card
            title="Live Instant Predictor"
            subtitle="Test your newly trained model immediately by modifying feature values below"
          >
            <form onSubmit={handleRunLivePrediction} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 max-h-56 overflow-y-auto p-1">
                {Object.keys(predictorInputs).map((feat) => (
                  <div key={feat} className="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-800/50 border text-xs space-y-1">
                    <label className="font-semibold text-gray-700 dark:text-gray-300 truncate block">
                      {feat}
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={predictorInputs[feat] ?? ''}
                      onChange={(e) =>
                        handleLiveInputChange(feat, parseFloat(e.target.value) || 0)
                      }
                      className="w-full bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded px-2.5 py-1 text-xs font-mono"
                    />
                  </div>
                ))}
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-gray-100 dark:border-gray-800">
                <button
                  type="submit"
                  disabled={isLivePredicting}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-brand-500 hover:bg-brand-600 disabled:opacity-50 text-white text-xs font-bold shadow-sm transition-all"
                >
                  {isLivePredicting ? (
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Zap className="w-3.5 h-3.5 fill-white" />
                  )}
                  <span>Evaluate Input</span>
                </button>

                {livePrediction && (
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-400">Output:</span>
                    <span className="px-4 py-1.5 rounded-lg bg-emerald-500 text-white font-mono font-extrabold text-base shadow-sm">
                      {livePrediction.prediction}
                    </span>
                  </div>
                )}
              </div>
            </form>
          </Card>
        </div>
      )}
    </div>
  );
};
