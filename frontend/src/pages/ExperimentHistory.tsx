import React, { useState, useEffect } from 'react';
import {
  History,
  Trash2,
  Eye,
  PlayCircle,
  Clock,
  Cpu,
  CheckCircle2,
  Filter,
  Search,
  Database,
  ArrowRight,
} from 'lucide-react';
import { Card } from '../components/Card';
import { ModelSummary } from '../types';
import {
  getExperiments,
  getExperimentDetails,
  deleteExperiment,
} from '../services/api';

interface ExperimentHistoryProps {
  onNavigateToPrediction?: (experimentId: string) => void;
  showToast: (type: 'success' | 'error' | 'info', message: string, title?: string) => void;
}

export const ExperimentHistory: React.FC<ExperimentHistoryProps> = ({
  onNavigateToPrediction,
  showToast,
}) => {
  const [experiments, setExperiments] = useState<ModelSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterAlgorithm, setFilterAlgorithm] = useState<string>('all');
  const [filterType, setFilterType] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');

  // Detail Modal State
  const [inspectModalOpen, setInspectModalOpen] = useState<boolean>(false);
  const [inspectDetails, setInspectDetails] = useState<any>(null);
  const [inspectLoading, setInspectLoading] = useState<boolean>(false);

  const fetchExperiments = async () => {
    try {
      setLoading(true);
      const data = await getExperiments();
      setExperiments(data);
    } catch (err) {
      showToast('error', 'Failed to load experiments.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExperiments();
  }, []);

  const handleInspect = async (id: string) => {
    try {
      setInspectLoading(true);
      setInspectModalOpen(true);
      const details = await getExperimentDetails(id);
      setInspectDetails(details);
    } catch (err) {
      showToast('error', 'Failed to load experiment details.');
    } finally {
      setInspectLoading(false);
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!window.confirm(`Delete experiment '${name}' and remove its trained artifact bundle?`)) {
      return;
    }
    try {
      await deleteExperiment(id);
      showToast('success', `Experiment '${name}' deleted.`);
      setExperiments(experiments.filter((exp) => exp.id !== id));
      if (inspectDetails?.id === id) {
        setInspectModalOpen(false);
      }
    } catch (err) {
      showToast('error', 'Failed to delete experiment.');
    }
  };

  // Filtered experiments
  const filtered = experiments.filter((exp) => {
    const matchAlgo = filterAlgorithm === 'all' || exp.algorithm === filterAlgorithm;
    const matchType = filterType === 'all' || exp.implementation_type === filterType;
    const matchSearch =
      exp.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      exp.dataset_name?.toLowerCase().includes(searchTerm.toLowerCase());
    return matchAlgo && matchType && matchSearch;
  });

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Search and Filters Header */}
      <div className="bg-white dark:bg-[#111827] p-5 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3 w-full md:w-auto">
          <History className="w-5 h-5 text-brand-500 shrink-0" />
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-gray-100">
              Experiment Run History
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {experiments.length} total experiments recorded in SQLite
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 w-full md:w-auto">
          {/* Search Input */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search experiments..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg pl-8 pr-3 py-1.5 text-xs text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-brand-500 focus:outline-none w-44"
            />
          </div>

          {/* Algorithm Filter */}
          <select
            value={filterAlgorithm}
            onChange={(e) => setFilterAlgorithm(e.target.value)}
            className="bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-900 dark:text-gray-100"
          >
            <option value="all">All Algorithms</option>
            <option value="linear_regression">Linear Regression</option>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="mlp">Multilayer Perceptron</option>
          </select>

          {/* Implementation Type Filter */}
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-900 dark:text-gray-100"
          >
            <option value="all">All Approaches</option>
            <option value="scratch">Scratch Only</option>
            <option value="library">Library Only</option>
          </select>
        </div>
      </div>

      {/* Experiments Table */}
      <Card>
        {filtered.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs divide-y divide-gray-200 dark:divide-gray-800">
              <thead className="bg-gray-50 dark:bg-gray-800/60 text-gray-500 dark:text-gray-400 font-semibold">
                <tr>
                  <th className="py-3 px-4">Experiment Name</th>
                  <th className="py-3 px-4">Dataset</th>
                  <th className="py-3 px-4">Algorithm</th>
                  <th className="py-3 px-4">Type</th>
                  <th className="py-3 px-4">Primary Metric</th>
                  <th className="py-3 px-4">Training Time</th>
                  <th className="py-3 px-4">Created</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800/60 font-sans">
                {filtered.map((exp) => {
                  const metricDisplay =
                    exp.metrics?.r2_score !== undefined
                      ? `R²: ${exp.metrics.r2_score}`
                      : exp.metrics?.accuracy !== undefined
                      ? `Acc: ${(exp.metrics.accuracy * 100).toFixed(1)}%`
                      : 'Completed';

                  return (
                    <tr
                      key={exp.id}
                      className="hover:bg-gray-50/70 dark:hover:bg-gray-800/30 transition-colors"
                    >
                      <td className="py-3 px-4 font-semibold text-gray-900 dark:text-gray-100">
                        {exp.name}
                      </td>
                      <td className="py-3 px-4 text-gray-600 dark:text-gray-300 flex items-center gap-1.5">
                        <Database className="w-3.5 h-3.5 text-gray-400" />
                        <span>{exp.dataset_name}</span>
                      </td>
                      <td className="py-3 px-4 capitalize text-gray-700 dark:text-gray-300">
                        {exp.algorithm.replace('_', ' ')}
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                            exp.implementation_type === 'scratch'
                              ? 'bg-purple-100 dark:bg-purple-950/60 text-purple-700 dark:text-purple-300'
                              : 'bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300'
                          }`}
                        >
                          {exp.implementation_type}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono font-bold text-emerald-600 dark:text-emerald-400">
                        {metricDisplay}
                      </td>
                      <td className="py-3 px-4 font-mono text-gray-500">
                        {exp.training_time_ms} ms
                      </td>
                      <td className="py-3 px-4 text-gray-400 text-[11px]">
                        {new Date(exp.created_at).toLocaleDateString()}
                      </td>
                      <td className="py-3 px-4 text-right space-x-1.5 whitespace-nowrap">
                        <button
                          onClick={() => handleInspect(exp.id)}
                          className="p-1.5 text-gray-400 hover:text-brand-500 hover:bg-brand-50 dark:hover:bg-brand-950/40 rounded-md transition-colors"
                          title="Inspect Experiment Details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        {onNavigateToPrediction && (
                          <button
                            onClick={() => onNavigateToPrediction(exp.id)}
                            className="p-1.5 text-gray-400 hover:text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-950/40 rounded-md transition-colors"
                            title="Predict with this Model"
                          >
                            <PlayCircle className="w-4 h-4" />
                          </button>
                        )}
                        <button
                          onClick={() => handleDelete(exp.id, exp.name)}
                          className="p-1.5 text-gray-400 hover:text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-950/40 rounded-md transition-colors"
                          title="Delete Experiment & Artifact"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-16 text-center text-xs text-gray-400">
            {searchTerm || filterAlgorithm !== 'all' || filterType !== 'all'
              ? 'No experiments match the filter criteria.'
              : 'No experiments recorded yet. Run a model in the Training Lab to view history.'}
          </div>
        )}
      </Card>

      {/* Inspect Experiment Modal */}
      {inspectModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-3 border-b border-gray-100 dark:border-gray-800">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                  <Eye className="w-5 h-5 text-brand-500" />
                  <span>{inspectDetails?.name || 'Experiment Details'}</span>
                </h3>
                <span className="text-[11px] text-gray-400 font-mono">
                  ID: {inspectDetails?.id}
                </span>
              </div>
              <button
                onClick={() => setInspectModalOpen(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 text-sm font-semibold"
              >
                ✕
              </button>
            </div>

            {inspectLoading ? (
              <div className="py-12 text-center text-gray-400 text-xs">Loading metadata...</div>
            ) : inspectDetails ? (
              <div className="space-y-4 text-xs">
                {/* Hyperparameters Card */}
                <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 space-y-2">
                  <span className="font-bold text-gray-800 dark:text-gray-200 uppercase tracking-wider text-[10px] block">
                    Hyperparameters & Architecture
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-[11px]">
                    <div>
                      <span className="text-gray-400">Algorithm:</span>{' '}
                      <span className="font-semibold text-gray-900 dark:text-gray-100">
                        {inspectDetails.algorithm}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Implementation:</span>{' '}
                      <span className="font-semibold text-gray-900 dark:text-gray-100">
                        {inspectDetails.implementation_type}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Task Type:</span>{' '}
                      <span className="font-semibold text-gray-900 dark:text-gray-100">
                        {inspectDetails.task_type}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Learning Rate:</span>{' '}
                      <span className="font-semibold text-gray-900 dark:text-gray-100">
                        {inspectDetails.hyperparameters?.learning_rate ?? 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Epochs:</span>{' '}
                      <span className="font-semibold text-gray-900 dark:text-gray-100">
                        {inspectDetails.hyperparameters?.epochs ?? 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Scaling:</span>{' '}
                      <span className="font-semibold text-gray-900 dark:text-gray-100">
                        {inspectDetails.hyperparameters?.scaling ?? 'standard'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Evaluation Metrics Card */}
                <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 space-y-2">
                  <span className="font-bold text-gray-800 dark:text-gray-200 uppercase tracking-wider text-[10px] block">
                    Evaluation Performance
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                    {Object.entries(inspectDetails.metrics || {}).map(([key, val]) => {
                      if (typeof val === 'number') {
                        return (
                          <div key={key}>
                            <span className="text-gray-400 capitalize">{key.replace('_', ' ')}:</span>{' '}
                            <span className="font-bold text-emerald-600 dark:text-emerald-400">
                              {val}
                            </span>
                          </div>
                        );
                      }
                      return null;
                    })}
                  </div>
                </div>

                {/* Selected Features */}
                <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 space-y-2">
                  <span className="font-bold text-gray-800 dark:text-gray-200 uppercase tracking-wider text-[10px] block">
                    Features ({inspectDetails.feature_names?.length || 0}) → Target ({inspectDetails.target_column})
                  </span>
                  <div className="flex flex-wrap gap-1.5 font-mono text-[10px]">
                    {inspectDetails.feature_names?.map((f: string) => (
                      <span
                        key={f}
                        className="px-2 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Actions inside modal */}
                <div className="pt-2 flex justify-end gap-2">
                  <button
                    onClick={() => setInspectModalOpen(false)}
                    className="px-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-xs font-semibold"
                  >
                    Close
                  </button>
                  {onNavigateToPrediction && (
                    <button
                      onClick={() => {
                        setInspectModalOpen(false);
                        onNavigateToPrediction(inspectDetails.id);
                      }}
                      className="px-4 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-xs font-semibold flex items-center gap-1.5 shadow-sm"
                    >
                      <PlayCircle className="w-3.5 h-3.5" />
                      <span>Predict with this Model</span>
                    </button>
                  )}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
};
