import React, { useEffect, useState, useRef } from 'react';
import {
  Database,
  Upload,
  FileText,
  AlertCircle,
  CheckCircle2,
  Table,
  BarChart,
  Grid,
  Hash,
  Type,
  Trash2,
  ArrowRight,
  RefreshCw,
} from 'lucide-react';
import { Card } from '../components/Card';
import { PlotlyChart } from '../components/PlotlyChart';
import { Dataset, DatasetPreview } from '../types';
import {
  getDatasets,
  getDatasetPreview,
  uploadDataset,
  deleteDataset,
} from '../services/api';

interface DatasetExplorerProps {
  onSelectDatasetForTraining?: (datasetId: string) => void;
  showToast: (type: 'success' | 'error' | 'info', message: string, title?: string) => void;
  initialOpenUpload?: boolean;
  onCloseUpload?: () => void;
}

export const DatasetExplorer: React.FC<DatasetExplorerProps> = ({
  onSelectDatasetForTraining,
  showToast,
  initialOpenUpload = false,
  onCloseUpload,
}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'preview' | 'stats' | 'correlation'>('preview');

  // Upload Modal State
  const [showUploadModal, setShowUploadModal] = useState<boolean>(initialOpenUpload);

  const closeModal = () => {
    setShowUploadModal(false);
    onCloseUpload?.();
  };

  useEffect(() => {
    if (initialOpenUpload) {
      setShowUploadModal(true);
    }
  }, [initialOpenUpload]);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState<string>('');
  const [uploadTask, setUploadTask] = useState<string>('');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const data = await getDatasets();
      setDatasets(data);
      if (data.length > 0 && !selectedId) {
        setSelectedId(data[0].id);
      }
    } catch (err: any) {
      showToast('error', err?.response?.data?.detail || 'Failed to load datasets.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    const fetchPreview = async () => {
      try {
        setPreviewLoading(true);
        const p = await getDatasetPreview(selectedId);
        setPreview(p);
      } catch (err: any) {
        showToast('error', err?.response?.data?.detail || 'Failed to load dataset preview.');
      } finally {
        setPreviewLoading(false);
      }
    };
    fetchPreview();
  }, [selectedId]);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) {
      showToast('error', 'Please choose a CSV file to upload.');
      return;
    }
    try {
      setIsUploading(true);
      const newDs = await uploadDataset(
        uploadFile,
        uploadName || undefined,
        uploadTask || undefined
      );
      showToast('success', `Dataset '${newDs.name}' uploaded successfully!`);
      closeModal();
      setUploadFile(null);
      setUploadName('');
      setUploadTask('');
      await fetchDatasets();
      setSelectedId(newDs.id);
    } catch (err: any) {
      let msg = err?.response?.data?.detail;
      if (!msg) {
        if (err?.code === 'ERR_NETWORK' || err?.message === 'Network Error') {
          msg = 'Cannot reach backend server. Please verify the Python API server is running on port 8000.';
        } else if (err?.response?.status === 500) {
          msg = 'Backend server temporarily unavailable or encountered an error. Please retry.';
        } else {
          msg = err?.message || 'Failed to upload dataset.';
        }
      }
      showToast('error', typeof msg === 'string' ? msg : JSON.stringify(msg), 'Upload Error');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!window.confirm(`Are you sure you want to delete dataset '${name}'?`)) return;
    try {
      await deleteDataset(id);
      showToast('success', `Dataset '${name}' deleted.`);
      const remaining = datasets.filter((d) => d.id !== id);
      setDatasets(remaining);
      if (remaining.length > 0) {
        setSelectedId(remaining[0].id);
      } else {
        setSelectedId('');
        setPreview(null);
      }
    } catch (err: any) {
      showToast('error', err?.response?.data?.detail || 'Failed to delete dataset.');
    }
  };

  const selectedDataset = datasets.find((d) => d.id === selectedId);

  // Build Correlation Heatmap Plotly Data
  const getCorrelationPlotlyData = () => {
    if (!preview?.correlation_matrix) return null;
    const cols = Object.keys(preview.correlation_matrix);
    const zValues = cols.map((row) =>
      cols.map((col) => preview.correlation_matrix![row][col] ?? 0.0)
    );

    return [
      {
        z: zValues,
        x: cols,
        y: cols,
        type: 'heatmap',
        colorscale: 'RdBu',
        reversescale: true,
        zmin: -1.0,
        zmax: 1.0,
        hoverongaps: false,
      },
    ];
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Top Bar with Dataset Selection & Upload Action */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-white dark:bg-[#111827] p-5 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm">
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <Database className="w-5 h-5 text-brand-500 shrink-0" />
          <div className="flex-1">
            <label className="block text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">
              Active Dataset
            </label>
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="w-full sm:w-72 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-gray-100 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:outline-none font-medium"
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} {d.is_builtin ? '(Built-in)' : ''}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full sm:w-auto">
          {selectedDataset && onSelectDatasetForTraining && (
            <button
              onClick={() => onSelectDatasetForTraining(selectedDataset.id)}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-50 dark:bg-emerald-950/60 border border-emerald-300 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100 dark:hover:bg-emerald-900/60 text-xs font-semibold transition-colors"
            >
              <span>Train on this Dataset</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}

          <button
            onClick={() => setShowUploadModal(true)}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-xs font-semibold transition-colors shadow-sm"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload CSV</span>
          </button>
        </div>
      </div>

      {/* Dataset Overview Summary Cards */}
      {preview && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
            <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider block">
              Total Rows
            </span>
            <span className="text-xl font-bold font-mono text-gray-900 dark:text-gray-100 mt-1 block">
              {preview.total_rows.toLocaleString()}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
            <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider block">
              Columns
            </span>
            <span className="text-xl font-bold font-mono text-gray-900 dark:text-gray-100 mt-1 block">
              {preview.total_columns}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
            <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider block">
              Numerical
            </span>
            <span className="text-xl font-bold font-mono text-blue-600 dark:text-blue-400 mt-1 block">
              {preview.numerical_features.length}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
            <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider block">
              Categorical
            </span>
            <span className="text-xl font-bold font-mono text-purple-600 dark:text-purple-400 mt-1 block">
              {preview.categorical_features.length}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
            <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider block">
              Duplicates
            </span>
            <span className="text-xl font-bold font-mono text-amber-600 dark:text-amber-400 mt-1 block">
              {preview.duplicate_rows}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 shadow-sm">
            <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider block">
              Missing Values
            </span>
            <span className="text-xl font-bold font-mono text-rose-600 dark:text-rose-400 mt-1 block">
              {Object.values(preview.missing_values).reduce((a, b) => a + b, 0)}
            </span>
          </div>
        </div>
      )}

      {/* Main Tabs Card */}
      <Card
        title={selectedDataset?.name || 'Dataset Preview'}
        subtitle={`Task: ${selectedDataset?.task_type.replace('_', ' ')} • File: ${selectedDataset?.filename}`}
        action={
          <div className="flex items-center gap-2">
            <div className="inline-flex p-1 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-xs">
              <button
                onClick={() => setActiveTab('preview')}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  activeTab === 'preview'
                    ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-xs'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900'
                }`}
              >
                Data Preview
              </button>
              <button
                onClick={() => setActiveTab('stats')}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  activeTab === 'stats'
                    ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-xs'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900'
                }`}
              >
                Statistical Summary
              </button>
              <button
                onClick={() => setActiveTab('correlation')}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  activeTab === 'correlation'
                    ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-xs'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900'
                }`}
              >
                Correlations
              </button>
            </div>

            {selectedDataset && !selectedDataset.is_builtin && (
              <button
                onClick={() => handleDelete(selectedDataset.id, selectedDataset.name)}
                className="p-1.5 text-gray-400 hover:text-rose-500 rounded-lg hover:bg-rose-50 dark:hover:bg-rose-950/30 transition-colors"
                title="Delete uploaded dataset"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>
        }
      >
        {previewLoading ? (
          <div className="py-20 text-center text-gray-400 flex items-center justify-center gap-2">
            <RefreshCw className="w-5 h-5 animate-spin text-brand-500" />
            <span>Parsing dataset preview...</span>
          </div>
        ) : preview ? (
          <>
            {/* Tab 1: Head Rows Table */}
            {activeTab === 'preview' && (
              <div className="overflow-x-auto max-h-[460px]">
                <table className="w-full text-left text-xs divide-y divide-gray-200 dark:divide-gray-800">
                  <thead className="bg-gray-50 dark:bg-gray-800/60 sticky top-0 z-10">
                    <tr>
                      <th className="py-2.5 px-3 font-semibold text-gray-500 uppercase tracking-wider">#</th>
                      {preview.columns.map((col) => {
                        const isNum = preview.numerical_features.includes(col);
                        const isTarget = col === selectedDataset?.target_column;
                        return (
                          <th key={col} className="py-2.5 px-3 font-semibold">
                            <div className="flex items-center gap-1.5">
                              <span className={isTarget ? 'text-brand-500 font-bold' : 'text-gray-800 dark:text-gray-200'}>
                                {col}
                              </span>
                              <span
                                className={`px-1.5 py-0.2 rounded text-[9px] font-mono uppercase font-bold ${
                                  isNum
                                    ? 'bg-blue-100 dark:bg-blue-950/80 text-blue-700 dark:text-blue-300'
                                    : 'bg-purple-100 dark:bg-purple-950/80 text-purple-700 dark:text-purple-300'
                                }`}
                              >
                                {isNum ? 'num' : 'cat'}
                              </span>
                            </div>
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 dark:divide-gray-800/60 font-mono">
                    {preview.head_rows.map((row, idx) => (
                      <tr key={idx} className="hover:bg-gray-50/70 dark:hover:bg-gray-800/40">
                        <td className="py-2 px-3 text-gray-400 font-sans">{idx + 1}</td>
                        {preview.columns.map((col) => (
                          <td key={col} className="py-2 px-3 text-gray-700 dark:text-gray-300 truncate max-w-[200px]">
                            {row[col] !== null && row[col] !== undefined ? String(row[col]) : (
                              <span className="text-rose-400 italic font-sans text-[11px]">NaN</span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Tab 2: Statistical Summaries */}
            {activeTab === 'stats' && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs divide-y divide-gray-200 dark:divide-gray-800">
                  <thead className="bg-gray-50 dark:bg-gray-800/60 font-semibold text-gray-500 dark:text-gray-400">
                    <tr>
                      <th className="py-2.5 px-3">Feature</th>
                      <th className="py-2.5 px-3">Type</th>
                      <th className="py-2.5 px-3">Missing</th>
                      <th className="py-2.5 px-3">Mean / Top</th>
                      <th className="py-2.5 px-3">Std Dev</th>
                      <th className="py-2.5 px-3">Min</th>
                      <th className="py-2.5 px-3">25%</th>
                      <th className="py-2.5 px-3">Median (50%)</th>
                      <th className="py-2.5 px-3">75%</th>
                      <th className="py-2.5 px-3">Max</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 dark:divide-gray-800/60 font-mono">
                    {Object.entries(preview.summary_statistics).map(([col, s]) => (
                      <tr key={col} className="hover:bg-gray-50/70 dark:hover:bg-gray-800/40">
                        <td className="py-2.5 px-3 font-medium text-gray-900 dark:text-gray-100 font-sans">
                          {col}
                        </td>
                        <td className="py-2.5 px-3 font-sans">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${
                              s.type === 'numerical'
                                ? 'bg-blue-100 dark:bg-blue-950/80 text-blue-700 dark:text-blue-300'
                                : 'bg-purple-100 dark:bg-purple-950/80 text-purple-700 dark:text-purple-300'
                            }`}
                          >
                            {s.type}
                          </span>
                        </td>
                        <td className="py-2.5 px-3">
                          {preview.missing_values[col] > 0 ? (
                            <span className="text-rose-500 font-bold">
                              {preview.missing_values[col]}
                            </span>
                          ) : (
                            <span className="text-gray-400">0</span>
                          )}
                        </td>
                        <td className="py-2.5 px-3">
                          {s.type === 'numerical' ? s.mean : `${s.top} (${s.freq})`}
                        </td>
                        <td className="py-2.5 px-3">{s.type === 'numerical' ? s.std : '—'}</td>
                        <td className="py-2.5 px-3">{s.type === 'numerical' ? s.min : '—'}</td>
                        <td className="py-2.5 px-3">{s.type === 'numerical' ? s['25%'] : '—'}</td>
                        <td className="py-2.5 px-3">{s.type === 'numerical' ? s['50%'] : '—'}</td>
                        <td className="py-2.5 px-3">{s.type === 'numerical' ? s['75%'] : '—'}</td>
                        <td className="py-2.5 px-3">{s.type === 'numerical' ? s.max : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Tab 3: Correlation Matrix Heatmap */}
            {activeTab === 'correlation' && (
              <div>
                {preview.correlation_matrix ? (
                  <div className="h-[460px]">
                    <PlotlyChart
                      data={getCorrelationPlotlyData() || []}
                      layout={{
                        title: 'Pearson Feature Correlation Heatmap',
                        xaxis: { tickangle: -45 },
                      }}
                    />
                  </div>
                ) : (
                  <div className="text-center py-16 text-gray-400 text-sm">
                    Correlation matrix requires at least 2 numerical features.
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-16 text-gray-400 text-sm">
            Select a dataset to view its profile.
          </div>
        )}
      </Card>

      {/* Upload CSV Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-gray-100 dark:border-gray-800">
              <h3 className="text-base font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <Upload className="w-5 h-5 text-brand-500" />
                <span>Upload New CSV Dataset</span>
              </h3>
              <button
                onClick={closeModal}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 text-sm font-semibold"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleUpload} className="space-y-4 text-sm">
              <div>
                <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
                  CSV File *
                </label>
                <input
                  type="file"
                  accept=".csv"
                  ref={fileInputRef}
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-gray-500 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-brand-50 file:text-brand-700 hover:file:bg-brand-100 dark:file:bg-brand-950/60 dark:file:text-brand-300 cursor-pointer"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
                  Dataset Display Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. Student Exam Performance"
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                  className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-brand-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
                  Task Type
                </label>
                <select
                  value={uploadTask}
                  onChange={(e) => setUploadTask(e.target.value)}
                  className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-brand-500 focus:outline-none"
                >
                  <option value="">Auto-Detect from Target</option>
                  <option value="regression">Regression (Continuous values)</option>
                  <option value="binary_classification">Binary Classification (2 classes)</option>
                  <option value="multiclass_classification">Multiclass Classification (3+ classes)</option>
                </select>
              </div>

              <div className="pt-3 flex justify-end gap-2 border-t border-gray-100 dark:border-gray-800">
                <button
                  type="button"
                  onClick={closeModal}
                  className="px-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 text-xs font-semibold transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isUploading}
                  className="px-4 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 disabled:opacity-50 text-white text-xs font-semibold transition-colors flex items-center gap-1.5"
                >
                  {isUploading && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                  <span>{isUploading ? 'Uploading...' : 'Confirm Upload'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
