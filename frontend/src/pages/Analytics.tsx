import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  Activity,
  Layers,
  Sparkles,
  RefreshCw,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
} from 'recharts';
import { Card } from '../components/Card';
import { PlotlyChart } from '../components/PlotlyChart';
import { ModelSummary, Dataset } from '../types';
import { getTrainedModels, getDatasets, getExperimentDetails } from '../services/api';

interface AnalyticsProps {
  showToast: (type: 'success' | 'error' | 'info', message: string, title?: string) => void;
}

export const Analytics: React.FC<AnalyticsProps> = ({ showToast }) => {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [modelDetails, setModelDetails] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [modelsData, datasetsData] = await Promise.all([
          getTrainedModels(),
          getDatasets(),
        ]);
        setModels(modelsData);
        setDatasets(datasetsData);
        if (modelsData.length > 0) {
          setSelectedModelId(modelsData[0].id);
        }
      } catch (err) {
        showToast('error', 'Failed to load analytics data.');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (!selectedModelId) return;
    const fetchDetails = async () => {
      try {
        const details = await getExperimentDetails(selectedModelId);
        setModelDetails(details);
      } catch (err) {
        showToast('error', 'Failed to load model diagnostics.');
      }
    };
    fetchDetails();
  }, [selectedModelId]);

  // Aggregate Benchmark Training Time Data
  const trainingTimeData = models.map((m) => ({
    name: m.name.length > 20 ? m.name.substring(0, 18) + '...' : m.name,
    duration_ms: m.training_time_ms,
    type: m.implementation_type,
    algorithm: m.algorithm,
  }));

  // Build ROC Curve Plotly trace if classification
  const getRocPlotlyData = () => {
    if (!modelDetails?.metrics?.roc_curve || modelDetails.metrics.roc_curve.length === 0) {
      return null;
    }
    const points = modelDetails.metrics.roc_curve;
    return [
      {
        x: points.map((p: any) => p.fpr),
        y: points.map((p: any) => p.tpr),
        type: 'scatter',
        mode: 'lines+markers',
        name: `ROC Curve (AUC = ${modelDetails.metrics.roc_auc ?? 'N/A'})`,
        line: { color: '#10B981', width: 2.5 },
      },
      {
        x: [0, 1],
        y: [0, 1],
        type: 'scatter',
        mode: 'lines',
        name: 'Chance Baseline',
        line: { dash: 'dash', color: '#9CA3AF' },
      },
    ];
  };

  // Build Loss Curve Data for selected model
  const lossData = (modelDetails?.loss_history || []).map((loss: number, idx: number) => ({
    epoch: idx + 1,
    train_loss: loss,
    val_loss: modelDetails.val_loss_history?.[idx],
  }));

  // Build Residuals Data for regression
  const residualData = (modelDetails?.metrics?.actual_vs_predicted || []).map((pt: any) => ({
    index: pt.index,
    residual: pt.residual,
    predicted: pt.predicted,
  }));

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Analytics Model Selector Header */}
      <div className="bg-white dark:bg-[#111827] p-5 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-brand-500" />
            <span>Interactive Model Diagnostics & Visualizations</span>
          </h2>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            Real model evaluations, residual errors, ROC curves, and benchmark durations
          </p>
        </div>

        <div className="w-full sm:w-auto">
          <select
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            className="w-full sm:w-72 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-gray-100 rounded-lg px-3 py-2 text-xs font-semibold focus:ring-2 focus:ring-brand-500"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} [{m.implementation_type}]
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Grid of Diagnostic Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 1. Training Time Benchmark Across All Models */}
        <Card
          title="Training Duration Benchmark"
          subtitle="Execution duration (ms) across all recorded models"
        >
          {trainingTimeData.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={trainingTimeData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-25} textAnchor="end" />
                  <YAxis tick={{ fontSize: 10 }} label={{ value: 'ms', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      borderColor: '#374151',
                      borderRadius: '8px',
                      fontSize: '11px',
                      color: '#F9FAFB',
                    }}
                  />
                  <Bar dataKey="duration_ms" name="Duration (ms)" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="py-16 text-center text-xs text-gray-400">No models trained yet.</div>
          )}
        </Card>

        {/* 2. Loss Curves */}
        <Card
          title="Convergence Dynamics (Loss Curve)"
          subtitle={`Iterative cost reduction for '${modelDetails?.name || 'Selected Model'}'`}
        >
          {lossData.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
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
                  {modelDetails?.val_loss_history?.length > 0 && (
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      name="Validation Loss"
                      stroke="#8B5CF6"
                      strokeWidth={2}
                      dot={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="py-16 text-center text-xs text-gray-400">
              Closed-form analytical fit (Single epoch loss).
            </div>
          )}
        </Card>

        {/* 3. Regression Residuals OR Classification ROC Curve */}
        {modelDetails?.task_type === 'regression' ? (
          <Card
            title="Residual Error Distribution"
            subtitle="Residual (Actual - Predicted) plotted against Predicted value"
          >
            {residualData.length > 0 ? (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={residualData.slice(0, 40)} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="index" tick={{ fontSize: 9 }} />
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
                    <Bar dataKey="residual" name="Residual (e)" fill="#F59E0B" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="py-16 text-center text-xs text-gray-400">No residual data available.</div>
            )}
          </Card>
        ) : (
          <Card
            title="Receiver Operating Characteristic (ROC)"
            subtitle={`True Positive Rate vs. False Positive Rate (AUC: ${modelDetails?.metrics?.roc_auc ?? 'N/A'})`}
          >
            {getRocPlotlyData() ? (
              <div className="h-64">
                <PlotlyChart
                  data={getRocPlotlyData() || []}
                  layout={{
                    xaxis: { title: 'False Positive Rate (FPR)', range: [-0.05, 1.05] },
                    yaxis: { title: 'True Positive Rate (TPR)', range: [-0.05, 1.05] },
                    margin: { t: 20, r: 20, l: 50, b: 50 },
                  }}
                />
              </div>
            ) : (
              <div className="py-16 text-center text-xs text-gray-400">
                ROC curve is plotted when class probabilities are available.
              </div>
            )}
          </Card>
        )}

        {/* 4. Dataset Overview Chart */}
        <Card
          title="Dataset Dimensions Overview"
          subtitle="Total rows and feature counts of registered datasets"
        >
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={datasets.map((d) => ({
                  name: d.name.length > 15 ? d.name.substring(0, 14) + '...' : d.name,
                  rows: d.total_rows,
                  columns: d.total_columns,
                }))}
                margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" />
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
                <Bar dataKey="rows" name="Sample Rows" fill="#10B981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="columns" name="Feature Columns" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
};
