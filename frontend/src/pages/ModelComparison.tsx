import React, { useState, useEffect } from 'react';
import {
  GitCompare,
  ArrowRight,
  TrendingUp,
  Clock,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  BarChart3,
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

        // Pre-select a scratch model for A and a compatible library model for B if available
        const scratch = data.find((m) => m.implementation_type === 'scratch');
        const library = scratch
          ? data.find((m) => m.implementation_type === 'library' && m.task_type === scratch.task_type) ||
            data.find((m) => m.implementation_type === 'library')
          : data.find((m) => m.implementation_type === 'library');

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

  const modelAObj = models.find((m) => m.id === selectedIdA);
  const modelBObj = models.find((m) => m.id === selectedIdB);
  const hasTaskMismatch = Boolean(modelAObj && modelBObj && modelAObj.task_type !== modelBObj.task_type);

  const handleRunComparison = async () => {
    if (!selectedIdA || !selectedIdB) {
      showToast('info', 'Please select two models to compare.');
      return;
    }
    if (selectedIdA === selectedIdB) {
      showToast('info', 'Please select two different models to compare.');
      return;
    }
    if (hasTaskMismatch) {
      showToast(
        'error',
        `Cannot compare models with differing tasks: '${modelAObj?.task_type}' vs '${modelBObj?.task_type}'.`
      );
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
    if (selectedIdA && selectedIdB && selectedIdA !== selectedIdB && !hasTaskMismatch) {
      handleRunComparison();
    }
  }, [selectedIdA, selectedIdB, hasTaskMismatch]);

  // Feature Importance Data Preparation
  const fiComp = comparison?.feature_importance_comparison;
  const fiA = fiComp?.model_a || comparison?.model_a?.feature_importance;
  const fiB = fiComp?.model_b || comparison?.model_b?.feature_importance;

  const isSupportedA = fiA?.supported === true;
  const isSupportedB = fiB?.supported === true;
  const isAnySupported = isSupportedA || isSupportedB;
  const isBothSupported = isSupportedA && isSupportedB;

  // Collect all unique feature names across both models
  const allFeatures = Array.from(
    new Set([
      ...(fiA?.rankings?.map((r) => r.feature) || []),
      ...(fiB?.rankings?.map((r) => r.feature) || []),
    ])
  );

  // Sort features descending by maximum importance between the two models
  const sortedFeatures = [...allFeatures].sort((a, b) => {
    const valA = Math.max(fiA?.scores?.[a] ?? 0, fiB?.scores?.[a] ?? 0);
    const valB = Math.max(fiA?.scores?.[b] ?? 0, fiB?.scores?.[b] ?? 0);
    return valB - valA;
  });

  // Prepare Plotly traces for horizontal grouped bar chart
  const getFeatureImportancePlotlyData = () => {
    if (!isAnySupported || sortedFeatures.length === 0) return null;

    // Invert feature array so highest importance feature is at the top of the horizontal bar chart
    const yFeatures = [...sortedFeatures].reverse();
    const traces: any[] = [];

    if (isSupportedA && fiA) {
      traces.push({
        y: yFeatures,
        x: yFeatures.map((f) => fiA.scores?.[f] ?? 0),
        name: `${comparison?.model_a.name} (${comparison?.model_a.implementation_type})`,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#8B5CF6' },
        hovertemplate: '<b>%{y}</b><br>Model A: %{x:.2f}%<extra></extra>',
      });
    }

    if (isSupportedB && fiB) {
      traces.push({
        y: yFeatures,
        x: yFeatures.map((f) => fiB.scores?.[f] ?? 0),
        name: `${comparison?.model_b.name} (${comparison?.model_b.implementation_type})`,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#3B82F6' },
        hovertemplate: '<b>%{y}</b><br>Model B: %{x:.2f}%<extra></extra>',
      });
    }

    return traces;
  };

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

        {hasTaskMismatch && (
          <div className="flex items-center gap-2 p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-xl text-xs text-amber-800 dark:text-amber-200">
            <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600 dark:text-amber-400" />
            <span>
              <strong>Task Mismatch:</strong> Model A is configured for{' '}
              <span className="font-semibold">{modelAObj?.task_type.replace('_', ' ')}</span> while Model B is configured for{' '}
              <span className="font-semibold">{modelBObj?.task_type.replace('_', ' ')}</span>. Both models must share the same task type to compare.
            </span>
          </div>
        )}

        <div className="flex items-center justify-between pt-2 border-t border-gray-100 dark:border-gray-800">
          <div className="text-xs text-gray-500 dark:text-gray-400">
            {selectedIdA && selectedIdB && !hasTaskMismatch && selectedIdA !== selectedIdB ? (
              <span className="text-emerald-600 dark:text-emerald-400 font-medium flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" /> Ready for side-by-side benchmark
              </span>
            ) : hasTaskMismatch ? (
              <span className="text-amber-600 dark:text-amber-400 font-medium">
                Select matching task types to compare
              </span>
            ) : (
              <span>Select two models above to compare</span>
            )}
          </div>
          <button
            type="button"
            onClick={handleRunComparison}
            disabled={!selectedIdA || !selectedIdB || selectedIdA === selectedIdB || hasTaskMismatch || comparing}
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-400 text-white font-semibold text-xs rounded-xl shadow-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-md cursor-pointer"
          >
            <GitCompare className={`w-4 h-4 ${comparing ? 'animate-spin' : ''}`} />
            <span>{comparing ? 'Comparing...' : 'Compare Models'}</span>
          </button>
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

          {/* Feature Importance Comparison Card */}
          <Card
            title="Feature Influence & Importance Comparison"
            subtitle="Side-by-side relative feature influence (%) based on learned model coefficients"
          >
            <div className="space-y-6">
              {/* Methodology Badges */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px] font-semibold text-gray-500 dark:text-gray-400">
                  Methodology:
                </span>
                <span className="px-2.5 py-1 rounded-md text-[10px] font-bold bg-purple-50 text-purple-700 dark:bg-purple-950/40 dark:text-purple-300 border border-purple-200 dark:border-purple-800/40">
                  {comparison.model_a.name}: {fiA?.method || 'N/A'}
                </span>
                <span className="px-2.5 py-1 rounded-md text-[10px] font-bold bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300 border border-blue-200 dark:border-blue-800/40">
                  {comparison.model_b.name}: {fiB?.method || 'N/A'}
                </span>
              </div>

              {/* Unsupported Model Explanations */}
              {(!isSupportedA || !isSupportedB) && (
                <div className="space-y-2">
                  {!isSupportedA && (
                    <div className="p-3.5 rounded-xl bg-amber-50/70 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800/40 text-xs text-amber-800 dark:text-amber-200 flex items-start gap-2.5">
                      <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-bold">{comparison.model_a.name} ({comparison.model_a.algorithm.toUpperCase()}): </span>
                        {fiA?.explanation || 'Direct feature importance is not supported for this model architecture.'}
                      </div>
                    </div>
                  )}
                  {!isSupportedB && (
                    <div className="p-3.5 rounded-xl bg-amber-50/70 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800/40 text-xs text-amber-800 dark:text-amber-200 flex items-start gap-2.5">
                      <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-bold">{comparison.model_b.name} ({comparison.model_b.algorithm.toUpperCase()}): </span>
                        {fiB?.explanation || 'Direct feature importance is not supported for this model architecture.'}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Plotly Grouped Horizontal Bar Chart & Ranked Table */}
              {isAnySupported && sortedFeatures.length > 0 ? (
                <div className="space-y-6">
                  <div>
                    <h4 className="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                      <BarChart3 className="w-4 h-4 text-brand-500" />
                      <span>Visual Feature Influence Breakdown</span>
                    </h4>
                    <div style={{ height: `${Math.max(260, sortedFeatures.length * 36)}px` }}>
                      <PlotlyChart
                        data={getFeatureImportancePlotlyData() || []}
                        layout={{
                          barmode: isBothSupported ? 'group' : 'relative',
                          xaxis: {
                            title: 'Relative Feature Importance (%)',
                            ticksuffix: '%',
                            automargin: true,
                          },
                          yaxis: {
                            automargin: true,
                          },
                          margin: { t: 20, r: 20, l: 110, b: 50 },
                          legend: {
                            orientation: 'h',
                            y: 1.15,
                            x: 0,
                          },
                        }}
                      />
                    </div>
                  </div>

                  {/* Ranked Comparison Table */}
                  <div>
                    <h4 className="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-2">
                      Ranked Feature Parity Table
                    </h4>
                    <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-800">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-gray-50 dark:bg-gray-800/60 font-semibold text-gray-500 dark:text-gray-400">
                          <tr>
                            <th className="py-2.5 px-4 w-16">Rank</th>
                            <th className="py-2.5 px-4">Feature Name</th>
                            <th className="py-2.5 px-4 text-purple-700 dark:text-purple-300 font-bold">
                              {comparison.model_a.name} ({comparison.model_a.implementation_type})
                            </th>
                            <th className="py-2.5 px-4 text-blue-700 dark:text-blue-300 font-bold">
                              {comparison.model_b.name} ({comparison.model_b.implementation_type})
                            </th>
                            <th className="py-2.5 px-4 font-bold">Delta (A - B)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100 dark:divide-gray-800 font-mono">
                          {sortedFeatures.map((feat, idx) => {
                            const scoreA = isSupportedA ? fiA?.scores?.[feat] : undefined;
                            const scoreB = isSupportedB ? fiB?.scores?.[feat] : undefined;
                            const delta = isBothSupported && scoreA !== undefined && scoreB !== undefined
                              ? fiComp?.differences?.[feat] ?? Number((scoreA - scoreB).toFixed(2))
                              : undefined;

                            return (
                              <tr key={feat} className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30">
                                <td className="py-2.5 px-4 text-gray-400 font-sans font-semibold">
                                  #{idx + 1}
                                </td>
                                <td className="py-2.5 px-4 font-sans font-medium text-gray-900 dark:text-gray-100">
                                  {feat}
                                </td>
                                <td className="py-2.5 px-4">
                                  {scoreA !== undefined ? (
                                    <div>
                                      <span className="font-bold text-purple-700 dark:text-purple-300">
                                        {scoreA.toFixed(2)}%
                                      </span>
                                      {fiA?.raw_coefficients?.[feat] !== undefined && (
                                        <span className="text-[10px] text-gray-400 block font-mono">
                                          coef: {fiA.raw_coefficients[feat]}
                                        </span>
                                      )}
                                    </div>
                                  ) : (
                                    <span className="text-gray-400 italic font-sans text-[11px]">
                                      N/A (MLP)
                                    </span>
                                  )}
                                </td>
                                <td className="py-2.5 px-4">
                                  {scoreB !== undefined ? (
                                    <div>
                                      <span className="font-bold text-blue-700 dark:text-blue-300">
                                        {scoreB.toFixed(2)}%
                                      </span>
                                      {fiB?.raw_coefficients?.[feat] !== undefined && (
                                        <span className="text-[10px] text-gray-400 block font-mono">
                                          coef: {fiB.raw_coefficients[feat]}
                                        </span>
                                      )}
                                    </div>
                                  ) : (
                                    <span className="text-gray-400 italic font-sans text-[11px]">
                                      N/A (MLP)
                                    </span>
                                  )}
                                </td>
                                <td className="py-2.5 px-4 font-bold">
                                  {delta !== undefined ? (
                                    <span
                                      className={
                                        delta >= 0
                                          ? 'text-emerald-600 dark:text-emerald-400'
                                          : 'text-rose-600 dark:text-rose-400'
                                      }
                                    >
                                      {delta > 0 ? '+' : ''}
                                      {delta.toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-gray-400 font-sans text-[11px]">—</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-8 text-center text-xs text-gray-400">
                  Neither model provides feature importance coefficients.
                </div>
              )}
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
