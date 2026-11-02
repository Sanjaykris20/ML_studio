import React, { useState, useEffect } from 'react';
import {
  PlayCircle,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Clock,
  RotateCcw,
  Sliders,
  Send,
  Zap,
} from 'lucide-react';
import { Card } from '../components/Card';
import {
  ModelSummary,
  ModelFeatureInfo,
  PredictionResult,
  PredictionHistoryItem,
} from '../types';
import {
  getTrainedModels,
  getModelFeatureInfo,
  predictModel,
  getModelPredictions,
} from '../services/api';

interface PredictionStudioProps {
  initialModelId?: string;
  showToast: (type: 'success' | 'error' | 'info', message: string, title?: string) => void;
}

export const PredictionStudio: React.FC<PredictionStudioProps> = ({
  initialModelId,
  showToast,
}) => {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>(initialModelId || '');
  const [featureInfo, setFeatureInfo] = useState<ModelFeatureInfo | null>(null);
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);

  // 1. Load models list
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoadingModels(true);
        const data = await getTrainedModels();
        setModels(data);
        if (data.length > 0) {
          const targetId = initialModelId && data.some((m) => m.id === initialModelId)
            ? initialModelId
            : data[0].id;
          setSelectedModelId(targetId);
        }
      } catch (err: any) {
        console.warn('Failed to fetch trained models:', err);
        const msg =
          err?.response?.data?.detail ||
          (err?.message === 'Network Error'
            ? 'Cannot connect to backend server. Make sure the API is running on port 8000.'
            : undefined);
        if (msg) {
          showToast('error', msg);
        }
      } finally {
        setLoadingModels(false);
      }
    };
    fetchModels();
  }, [initialModelId]);

  // 2. When selected model changes, load its feature schema & predictions history
  useEffect(() => {
    if (!selectedModelId) return;
    const fetchSchema = async () => {
      try {
        const info = await getModelFeatureInfo(selectedModelId);
        setFeatureInfo(info);

        // Populate initial defaults using feature stats (mean or default value)
        const defaults: Record<string, any> = {};
        for (const feat of info.feature_names) {
          const stat = info.feature_stats[feat];
          if (stat) {
            if (stat.type === 'numerical') {
              defaults[feat] = stat.mean !== undefined ? round(stat.mean, 2) : 0;
            } else {
              defaults[feat] = stat.default || (stat.unique_values?.[0] ?? '');
            }
          } else {
            defaults[feat] = 0;
          }
        }
        setInputValues(defaults);
        setPredictionResult(null);

        // Fetch past predictions
        const past = await getModelPredictions(selectedModelId);
        setHistory(past);
      } catch (err: any) {
        showToast('error', err?.response?.data?.detail || 'Failed to load model features.');
      }
    };
    fetchSchema();
  }, [selectedModelId]);

  const round = (val: number, decimals: number) => {
    const factor = Math.pow(10, decimals);
    return Math.round(val * factor) / factor;
  };

  const handleInputChange = (feature: string, value: any) => {
    setInputValues((prev) => ({
      ...prev,
      [feature]: value,
    }));
  };

  const handleResetToDefaults = () => {
    if (!featureInfo) return;
    const defaults: Record<string, any> = {};
    for (const feat of featureInfo.feature_names) {
      const stat = featureInfo.feature_stats[feat];
      if (stat?.type === 'numerical') {
        defaults[feat] = round(stat.mean ?? 0, 2);
      } else {
        defaults[feat] = stat?.default || (stat?.unique_values?.[0] ?? '');
      }
    }
    setInputValues(defaults);
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedModelId) return;

    try {
      setIsPredicting(true);
      const res = await predictModel(selectedModelId, inputValues);
      setPredictionResult(res);
      showToast('success', `Predicted: ${res.prediction}`, 'Inference Complete');

      // Refresh history
      const past = await getModelPredictions(selectedModelId);
      setHistory(past);
    } catch (err: any) {
      showToast('error', err?.response?.data?.detail || 'Prediction failed.');
    } finally {
      setIsPredicting(false);
    }
  };

  if (!loadingModels && models.length === 0) {
    return (
      <div className="space-y-6 animate-in fade-in duration-300">
        <div className="text-center py-16 bg-white dark:bg-[#111827] rounded-xl border border-gray-200 dark:border-gray-800 p-8 space-y-4 shadow-sm">
          <div className="w-14 h-14 rounded-full bg-brand-50 dark:bg-brand-950/60 text-brand-500 mx-auto flex items-center justify-center">
            <PlayCircle className="w-7 h-7" />
          </div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100">
            No Trained Models Available Yet
          </h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 max-w-md mx-auto leading-relaxed">
            Train your first machine learning algorithm in the <strong>Model Training Lab</strong> or use the <strong>Auto Clean & Train Wizard</strong>. Once trained, your model and its fitted preprocessing pipeline will automatically appear here for live predictions!
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Model Selection Bar */}
      <div className="bg-white dark:bg-[#111827] p-5 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <PlayCircle className="w-5 h-5 text-brand-500 shrink-0" />
          <div className="flex-1">
            <label className="block text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">
              Select Trained Model for Inference
            </label>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full sm:w-80 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-gray-100 rounded-lg px-3 py-2 text-sm font-medium focus:ring-2 focus:ring-brand-500 focus:outline-none"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.implementation_type} • {m.task_type.replace('_', ' ')})
                </option>
              ))}
            </select>
          </div>
        </div>

        {featureInfo && (
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleResetToDefaults}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 text-xs font-semibold text-gray-600 dark:text-gray-300 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Fill Dataset Means</span>
            </button>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Dynamic Feature Form (7 cols) */}
        <div className="lg:col-span-7">
          <form onSubmit={handlePredict}>
            <Card
              title="Feature Inputs"
              subtitle={`Model inputs automatically configured (${featureInfo?.feature_names.length || 0} features)`}
              action={
                <button
                  type="submit"
                  disabled={isPredicting || !selectedModelId}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 disabled:opacity-50 text-white text-xs font-bold shadow-sm transition-all"
                >
                  {isPredicting ? (
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Zap className="w-3.5 h-3.5 fill-white" />
                  )}
                  <span>{isPredicting ? 'Evaluating...' : 'Predict Now'}</span>
                </button>
              }
            >
              {featureInfo ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[500px] overflow-y-auto pr-2">
                  {featureInfo.feature_names.map((feat) => {
                    const stat = featureInfo.feature_stats[feat];
                    const isNum = stat?.type === 'numerical';

                    return (
                      <div
                        key={feat}
                        className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 border border-gray-100 dark:border-gray-800 space-y-1.5"
                      >
                        <div className="flex items-center justify-between text-xs">
                          <label className="font-semibold text-gray-800 dark:text-gray-200 truncate">
                            {feat}
                          </label>
                          {isNum && stat && (
                            <span className="text-[10px] text-gray-400 font-mono">
                              [{stat.min} ~ {stat.max}]
                            </span>
                          )}
                        </div>

                        {isNum ? (
                          <input
                            type="number"
                            step="any"
                            value={inputValues[feat] ?? ''}
                            onChange={(e) =>
                              handleInputChange(feat, parseFloat(e.target.value) || 0)
                            }
                            className="w-full bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md px-3 py-1.5 text-xs text-gray-900 dark:text-gray-100 font-mono focus:ring-1 focus:ring-brand-500 focus:outline-none"
                            required
                          />
                        ) : stat?.unique_values ? (
                          <select
                            value={inputValues[feat] ?? ''}
                            onChange={(e) => handleInputChange(feat, e.target.value)}
                            className="w-full bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md px-3 py-1.5 text-xs text-gray-900 dark:text-gray-100 focus:ring-1 focus:ring-brand-500 focus:outline-none"
                          >
                            {stat.unique_values.map((val: string) => (
                              <option key={val} value={val}>
                                {val}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="text"
                            value={inputValues[feat] ?? ''}
                            onChange={(e) => handleInputChange(feat, e.target.value)}
                            className="w-full bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md px-3 py-1.5 text-xs text-gray-900 dark:text-gray-100 focus:ring-1 focus:ring-brand-500 focus:outline-none"
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="py-16 text-center text-xs text-gray-400">
                  Select a trained model above to build input controls.
                </div>
              )}
            </Card>
          </form>
        </div>

        {/* Right Column: Prediction Output & Probability Meters (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          <Card title="Prediction Output" subtitle="Instantaneous inference result">
            {predictionResult ? (
              <div className="space-y-5 py-2">
                <div className="p-6 rounded-2xl bg-gradient-to-br from-gray-900 to-gray-800 dark:from-gray-950 dark:to-gray-900 text-white text-center shadow-lg border border-gray-700">
                  <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-widest block mb-1">
                    {predictionResult.task_type === 'regression' ? 'Estimated Value' : 'Predicted Class'}
                  </span>
                  <div className="text-4xl font-extrabold font-mono text-white tracking-tight">
                    {predictionResult.prediction}
                  </div>
                  <div className="text-xs text-gray-400 mt-2 font-mono">
                    Target: {featureInfo?.target_column}
                  </div>
                </div>

                {/* Class Probabilities Distribution */}
                {predictionResult.probabilities && (
                  <div className="space-y-3">
                    <span className="text-xs font-bold text-gray-800 dark:text-gray-200 block">
                      Confidence & Probabilities
                    </span>
                    <div className="space-y-2">
                      {Object.entries(predictionResult.probabilities).map(([cls, prob]) => {
                        const pct = Math.round(prob * 100);
                        return (
                          <div key={cls} className="space-y-1">
                            <div className="flex justify-between text-xs font-mono">
                              <span className="text-gray-600 dark:text-gray-300 font-semibold">
                                {cls}
                              </span>
                              <span className="text-emerald-600 dark:text-emerald-400 font-bold">
                                {pct}% ({prob})
                              </span>
                            </div>
                            <div className="w-full h-2 rounded-full bg-gray-100 dark:bg-gray-800 overflow-hidden">
                              <div
                                className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-16 text-center text-xs text-gray-400 flex flex-col items-center justify-center gap-2">
                <Sparkles className="w-6 h-6 text-gray-400" />
                <span>Submit input values to run real-time inference.</span>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Historical Predictions Table */}
      {history.length > 0 && (
        <Card
          title="Prediction History"
          subtitle={`Previous inference logs for this model (${history.length} records)`}
        >
          <div className="overflow-x-auto max-h-60">
            <table className="w-full text-left text-xs">
              <thead className="bg-gray-50 dark:bg-gray-800/60 text-gray-500 dark:text-gray-400 font-semibold">
                <tr>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Predicted Value</th>
                  <th className="py-2.5 px-3">Probabilities</th>
                  <th className="py-2.5 px-3">Sample Input Values</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800 font-mono">
                {history.map((item) => (
                  <tr key={item.id} className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30">
                    <td className="py-2 px-3 text-gray-400 font-sans text-[11px]">
                      {new Date(item.created_at).toLocaleTimeString()}
                    </td>
                    <td className="py-2 px-3 font-bold text-emerald-600 dark:text-emerald-400">
                      {item.predicted_value}
                    </td>
                    <td className="py-2 px-3 text-gray-600 dark:text-gray-300 text-[11px]">
                      {item.probabilities
                        ? Object.entries(item.probabilities)
                            .map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`)
                            .join(', ')
                        : 'N/A'}
                    </td>
                    <td className="py-2 px-3 text-gray-500 truncate max-w-xs text-[11px]">
                      {JSON.stringify(item.input_values)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
};
