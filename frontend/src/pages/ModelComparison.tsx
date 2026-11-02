import React, { useState, useEffect } from 'react';
import {
  GitCompare,
  ArrowRight,
  TrendingUp,
  Clock,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';
import { Card } from '../components/Card';
import { PlotlyChart } from '../components/PlotlyChart';
import {
  ModelSummary,
  ModelComparisonData,
} from '../types';
import { getTrainedModels, compareModels } from '../services/api';

interface ModelComparisonProps {
  showToast: (type: 'success' | 'error' | 'info', message: string, title?: string) => void;
}

export const ModelComparison: React.FC<ModelComparisonProps> = ({ showToast }) => {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedIdA, setSelectedIdA] = useState<string>('');
  const [selectedIdB, setSelectedIdB] = useState<string>('');
  const [comparison, setComparison] = useState<ModelComparisonData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [comparing, setComparing] = useState<boolean>(false);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const data = await getTrainedModels();
        setModels(data);

        // Pre-select a scratch model for A and a library model for B if available
        const scratch = data.find((m) => m.implementation_type === 'scratch');
        const library = data.find((m) => m.implementation_type === 'library');

        if (scratch) setSelectedIdA(scratch.id);
        if (library) setSelectedIdB(library.id);
        else if (data.length > 1) setSelectedIdB(data[1].id);
      } catch (err) {
        showToast('error', 'Failed to load models for comparison.');
      } finally {
        setLoading(false);
      }
    };
    fetchModels();
  }, []);

  const handleRunComparison = async () => {
    if (!selectedIdA || !selectedIdB) {
      showToast('info', 'Please select two models to compare.');
      return;
    }
    if (selectedIdA === selectedIdB) {
      showToast('info', 'Please select two different models to compare.');
      return;
    }

    try {
      setComparing(true);
      const res = await compareModels(selectedIdA, selectedIdB);
      setComparison(res);
      if (!res.comparison_valid) {
        showToast('error', res.notes || 'Incompatible models cannot be compared.');
      } else {
        showToast('success', 'Side-by-side benchmark loaded.');
      }
    } catch (err: any) {
      showToast('error', err?.response?.data?.detail || 'Comparison failed.');
    } finally {
      setComparing(false);
    }
  };

  useEffect(() => {
    if (selectedIdA && selectedIdB && selectedIdA !== selectedIdB) {
      handleRunComparison();
    }
  }, [selectedIdA, selectedIdB]);

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Model Pair Selectors */}
      <div className="p-6 rounded-2xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm space-y-4">
        <div className="flex items-center gap-2 text-sm font-bold text-gray-900 dark:text-gray-100">
          <GitCompare className="w-5 h-5 text-brand-500" />
          <span>Side-by-Side Benchmark Workbench</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Model A Selector */}
          <div className="p-4 rounded-xl bg-purple-50/50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-800/40">
            <span className="text-[10px] font-bold text-purple-700 dark:text-purple-300 uppercase tracking-widest block mb-1">
              Model A (Scratch Implementation)
            </span>
            <select
              value={selectedIdA}
              onChange={(e) => setSelectedIdA(e.target.value)}
              className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-2 text-xs font-semibold text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-purple-500"
            >
              <option value="">Select Model A</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} [{m.implementation_type}] ({m.task_type.replace('_', ' ')})
                </option>
              ))}
            </select>
          </div>

          {/* Model B Selector */}
          <div className="p-4 rounded-xl bg-blue-50/50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800/40">
            <span className="text-[10px] font-bold text-blue-700 dark:text-blue-300 uppercase tracking-widest block mb-1">
              Model B (Library Implementation)
            </span>
            <select
              value={selectedIdB}
              onChange={(e) => setSelectedIdB(e.target.value)}
              className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-2 text-xs font-semibold text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select Model B</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} [{m.implementation_type}] ({m.task_type.replace('_', ' ')})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Comparison Results Card */}
      {comparison && comparison.comparison_valid ? (
        <div className="space-y-6">
          {/* Metrics Comparison Table */}
          <Card
            title="Performance Metrics Parity"
            subtitle={`Task: ${comparison.task_type.replace('_', ' ')} • Dual evaluation on identical test split`}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-gray-50 dark:bg-gray-800/60 font-semibold text-gray-500 dark:text-gray-400">
                  <tr>
                    <th className="py-3 px-4">Evaluation Metric</th>
                    <th className="py-3 px-4 text-purple-700 dark:text-purple-300 font-bold">
                      {comparison.model_a.name} ({comparison.model_a.implementation_type})
                    </th>
                    <th className="py-3 px-4 text-blue-700 dark:text-blue-300 font-bold">
                      {comparison.model_b.name} ({comparison.model_b.implementation_type})
                    </th>
                    <th className="py-3 px-4 font-bold">Delta (A - B)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800 font-mono">
                  {comparison.task_type === 'regression' ? (
                    <>
                      <tr className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30">
                        <td className="py-3 px-4 font-sans font-medium text-gray-900 dark:text-gray-100">
                          R² Score (Coefficient of Determination)
                        </td>
                        <td className="py-3 px-4 font-bold text-gray-800 dark:text-gray-200">
                          {comparison.model_a.metrics?.r2_score ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4 font-bold text-gray-800 dark:text-gray-200">
                          {comparison.model_b.metrics?.r2_score ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4 font-bold">
                          <span
                            className={
                              (comparison.metric_differences.r2_score ?? 0) >= 0
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400'
                            }
                          >
                            {(comparison.metric_differences.r2_score ?? 0) > 0 ? '+' : ''}
                            {comparison.metric_differences.r2_score ?? 0}
                          </span>
                        </td>
                      </tr>

                      <tr className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30">
                        <td className="py-3 px-4 font-sans font-medium text-gray-900 dark:text-gray-100">
                          Mean Squared Error (MSE)
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_a.metrics?.mse ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_b.metrics?.mse ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={
                              (comparison.metric_differences.mse ?? 0) <= 0
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400'
                            }
                          >
                            {(comparison.metric_differences.mse ?? 0) > 0 ? '+' : ''}
                            {comparison.metric_differences.mse ?? 0}
                          </span>
                        </td>
                      </tr>

                      <tr className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30">
                        <td className="py-3 px-4 font-sans font-medium text-gray-900 dark:text-gray-100">
                          Root Mean Squared Error (RMSE)
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_a.metrics?.rmse ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_b.metrics?.rmse ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={
                              (comparison.metric_differences.rmse ?? 0) <= 0
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400'
                            }
                          >
                            {(comparison.metric_differences.rmse ?? 0) > 0 ? '+' : ''}
                            {comparison.metric_differences.rmse ?? 0}
                          </span>
                        </td>
                      </tr>

                      <tr className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30">
                        <td className="py-3 px-4 font-sans font-medium text-gray-900 dark:text-gray-100">
                          Mean Absolute Error (MAE)
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_a.metrics?.mae ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_b.metrics?.mae ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={
                              (comparison.metric_differences.mae ?? 0) <= 0
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400'
                            }
                          >
                            {(comparison.metric_differences.mae ?? 0) > 0 ? '+' : ''}
                            {comparison.metric_differences.mae ?? 0}
                          </span>
                        </td>
                      </tr>
                    </>
                  ) : (
                    <>
                      <tr className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30">
                        <td className="py-3 px-4 font-sans font-medium text-gray-900 dark:text-gray-100">
                          Classification Accuracy
                        </td>
                        <td className="py-3 px-4 font-bold text-gray-800 dark:text-gray-200">
                          {comparison.model_a.metrics?.accuracy
                            ? `${(comparison.model_a.metrics.accuracy * 100).toFixed(2)}%`
                            : 'N/A'}
                        </td>
                        <td className="py-3 px-4 font-bold text-gray-800 dark:text-gray-200">
                          {comparison.model_b.metrics?.accuracy
                            ? `${(comparison.model_b.metrics.accuracy * 100).toFixed(2)}%`
                            : 'N/A'}
                        </td>
                        <td className="py-3 px-4 font-bold">
                          <span
                            className={
                              (comparison.metric_differences.accuracy ?? 0) >= 0
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400'
                            }
                          >
                            {(comparison.metric_differences.accuracy ?? 0) > 0 ? '+' : ''}
                            {comparison.metric_differences.accuracy
                              ? `${(comparison.metric_differences.accuracy * 100).toFixed(2)}%`
                              : '0%'}
                          </span>
                        </td>
                      </tr>

                      <tr className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30">
                        <td className="py-3 px-4 font-sans font-medium text-gray-900 dark:text-gray-100">
                          F1 Score (Weighted)
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_a.metrics?.f1_score ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_b.metrics?.f1_score ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={
                              (comparison.metric_differences.f1_score ?? 0) >= 0
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400'
                            }
                          >
                            {(comparison.metric_differences.f1_score ?? 0) > 0 ? '+' : ''}
                            {comparison.metric_differences.f1_score ?? 0}
                          </span>
                        </td>
                      </tr>

                      <tr className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30">
                        <td className="py-3 px-4 font-sans font-medium text-gray-900 dark:text-gray-100">
                          ROC-AUC Score
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_a.metrics?.roc_auc ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                          {comparison.model_b.metrics?.roc_auc ?? 'N/A'}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={
                              (comparison.metric_differences.roc_auc ?? 0) >= 0
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400'
                            }
                          >
                            {(comparison.metric_differences.roc_auc ?? 0) > 0 ? '+' : ''}
                            {comparison.metric_differences.roc_auc ?? 0}
                          </span>
                        </td>
                      </tr>
                    </>
                  )}

                  {/* Training Duration Comparison */}
                  <tr className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30 bg-gray-50/30 dark:bg-gray-800/10">
                    <td className="py-3 px-4 font-sans font-medium text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
                      <Clock className="w-3.5 h-3.5 text-gray-400" />
                      <span>Training Duration</span>
                    </td>
                    <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                      {comparison.model_a.training_time_ms} ms
                    </td>
                    <td className="py-3 px-4 text-gray-800 dark:text-gray-200">
                      {comparison.model_b.training_time_ms} ms
                    </td>
                    <td className="py-3 px-4">
                      <span>
                        {(comparison.metric_differences.training_time_ms_delta ?? 0) > 0 ? '+' : ''}
                        {comparison.metric_differences.training_time_ms_delta ?? 0} ms
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      ) : (
        <Card className="p-12 text-center text-xs text-gray-400">
          Select two compatible models above to generate side-by-side metric diffs and comparison plots.
        </Card>
      )}
    </div>
  );
};
