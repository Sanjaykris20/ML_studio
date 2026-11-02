import React, { useEffect, useState } from 'react';
import {
  Database,
  Cpu,
  PlayCircle,
  GitCompare,
  BarChart3,
  CheckCircle2,
  Clock,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  TrendingUp,
  Wand2,
  Upload,
} from 'lucide-react';
import { Card } from '../components/Card';
import { MetricCard } from '../components/MetricCard';
import { DashboardStats, SystemHealth } from '../types';
import { getDashboardStats, getSystemHealth } from '../services/api';

interface DashboardProps {
  setActivePage: (page: string) => void;
  onUploadClick?: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ setActivePage, onUploadClick }) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsData, healthData] = await Promise.all([
          getDashboardStats(),
          getSystemHealth(),
        ]);
        setStats(statsData);
        setHealth(healthData);
      } catch (err) {
        console.error('Error loading dashboard stats:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-gray-900 via-gray-800 to-emerald-950 border border-gray-800 p-8 text-white shadow-lg">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs font-semibold mb-4">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Interactive Machine Learning Workbench</span>
          </div>
          <h2 className="text-3xl font-extrabold tracking-tight sm:text-4xl">
            Train, Evaluate & Compare ML from Scratch vs. Reference Libraries
          </h2>
          <p className="mt-3 text-sm text-gray-300 leading-relaxed max-w-2xl">
            Inspect data without leakage, train analytical Linear Regression and Logistic Regression in NumPy,
            build custom PyTorch autograd Multilayer Perceptrons, and compare fair parity side-by-side with Scikit-learn.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <button
              onClick={() => setActivePage('auto_train')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-brand-500 hover:from-emerald-600 hover:to-brand-600 text-white font-bold text-sm transition-all shadow-lg shadow-emerald-500/25 ring-2 ring-emerald-400/30"
            >
              <Wand2 className="w-4 h-4" />
              <span>Auto Clean & Train Wizard</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => setActivePage('training')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 text-white font-medium text-sm backdrop-blur-sm transition-all border border-white/10"
            >
              <Cpu className="w-4 h-4" />
              <span>Manual Training Lab</span>
            </button>
            <button
              onClick={() => setActivePage('datasets')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 text-white font-medium text-sm backdrop-blur-sm transition-all border border-white/10"
            >
              <Database className="w-4 h-4" />
              <span>Explore Datasets</span>
            </button>
            {onUploadClick && (
              <button
                onClick={onUploadClick}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/15 hover:bg-white/25 text-white font-medium text-sm backdrop-blur-sm transition-all border border-white/20"
              >
                <Upload className="w-4 h-4 text-emerald-400" />
                <span>Upload CSV Dataset</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Top Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <MetricCard
          label="Available Datasets"
          value={stats?.total_datasets ?? 0}
          subValue="3 Built-in benchmark datasets"
          icon={Database}
          accentColor="blue"
        />
        <MetricCard
          label="Total Experiments"
          value={stats?.total_experiments ?? 0}
          subValue="Persisted in SQLite"
          icon={Cpu}
          accentColor="emerald"
        />
        <MetricCard
          label="Scratch Implementations"
          value={stats?.scratch_experiments_count ?? 0}
          subValue="NumPy & PyTorch autograd"
          icon={TrendingUp}
          accentColor="purple"
        />
        <MetricCard
          label="Inference Predictions"
          value={stats?.total_predictions ?? 0}
          subValue="Real-time saved predictions"
          icon={PlayCircle}
          accentColor="amber"
        />
      </div>

      {/* Quick Launch Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div
          onClick={() => setActivePage('training')}
          className="cursor-pointer group p-6 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 hover:border-brand-500/50 dark:hover:border-brand-500/50 shadow-sm transition-all hover:shadow-md"
        >
          <div className="w-10 h-10 rounded-lg bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
            <Cpu className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 flex items-center justify-between">
            <span>Model Training Lab</span>
            <ArrowRight className="w-4 h-4 text-gray-400 group-hover:translate-x-1 transition-transform" />
          </h3>
          <p className="mt-1.5 text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
            Configure learning rates, batch sizes, and hidden layers. Train Linear, Logistic, and MLP models with real-time loss curves.
          </p>
        </div>

        <div
          onClick={() => setActivePage('prediction')}
          className="cursor-pointer group p-6 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 hover:border-blue-500/50 dark:hover:border-blue-500/50 shadow-sm transition-all hover:shadow-md"
        >
          <div className="w-10 h-10 rounded-lg bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
            <PlayCircle className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 flex items-center justify-between">
            <span>Prediction Studio</span>
            <ArrowRight className="w-4 h-4 text-gray-400 group-hover:translate-x-1 transition-transform" />
          </h3>
          <p className="mt-1.5 text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
            Input custom feature values via auto-generated forms. View instantaneous continuous predictions or class probabilities.
          </p>
        </div>

        <div
          onClick={() => setActivePage('comparison')}
          className="cursor-pointer group p-6 rounded-xl bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 hover:border-purple-500/50 dark:hover:border-purple-500/50 shadow-sm transition-all hover:shadow-md"
        >
          <div className="w-10 h-10 rounded-lg bg-purple-50 dark:bg-purple-950/50 text-purple-600 dark:text-purple-400 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
            <GitCompare className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 flex items-center justify-between">
            <span>Model Comparison</span>
            <ArrowRight className="w-4 h-4 text-gray-400 group-hover:translate-x-1 transition-transform" />
          </h3>
          <p className="mt-1.5 text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
            Direct side-by-side benchmark of Scratch vs. Scikit-learn models. Compare R², MSE, Accuracy, and duration deltas.
          </p>
        </div>
      </div>

      {/* Recent Experiments & Architecture Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card
            title="Recent Experiment Runs"
            subtitle="Latest trained models stored in local database"
            action={
              <button
                onClick={() => setActivePage('history')}
                className="text-xs font-semibold text-brand-600 dark:text-brand-400 hover:underline"
              >
                View All ({stats?.total_experiments ?? 0})
              </button>
            }
          >
            {stats?.recent_experiments && stats.recent_experiments.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-gray-50 dark:bg-gray-800/40 text-gray-500 dark:text-gray-400 border-b border-gray-100 dark:border-gray-800">
                    <tr>
                      <th className="py-2.5 px-3">Experiment</th>
                      <th className="py-2.5 px-3">Algorithm</th>
                      <th className="py-2.5 px-3">Type</th>
                      <th className="py-2.5 px-3">Primary Metric</th>
                      <th className="py-2.5 px-3">Training Time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 dark:divide-gray-800/60">
                    {stats.recent_experiments.map((exp) => {
                      const metricKey =
                        exp.metrics?.r2_score !== undefined
                          ? `R²: ${exp.metrics.r2_score}`
                          : exp.metrics?.accuracy !== undefined
                          ? `Acc: ${(exp.metrics.accuracy * 100).toFixed(1)}%`
                          : 'Trained';

                      return (
                        <tr
                          key={exp.id}
                          className="hover:bg-gray-50/60 dark:hover:bg-gray-800/30 transition-colors"
                        >
                          <td className="py-3 px-3 font-medium text-gray-900 dark:text-gray-200">
                            {exp.name}
                          </td>
                          <td className="py-3 px-3 text-gray-600 dark:text-gray-300 capitalize">
                            {exp.algorithm.replace('_', ' ')}
                          </td>
                          <td className="py-3 px-3">
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
                          <td className="py-3 px-3 font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                            {metricKey}
                          </td>
                          <td className="py-3 px-3 text-gray-500 dark:text-gray-400 font-mono">
                            {exp.training_time_ms} ms
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-10 text-gray-400 dark:text-gray-500 text-sm">
                No experiments recorded yet. Click "Launch Training Lab" to run your first model!
              </div>
            )}
          </Card>
        </div>

        {/* Engine Specs & Safety Status */}
        <div className="space-y-5">
          <Card title="System Diagnostics" subtitle="Local computational backend specs">
            <div className="space-y-3 text-xs">
              <div className="flex items-center justify-between py-1.5 border-b border-gray-100 dark:border-gray-800">
                <span className="text-gray-500 dark:text-gray-400">Engine Status</span>
                <span className="flex items-center gap-1.5 font-semibold text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  {health?.status === 'healthy' ? 'Operational' : 'Initializing'}
                </span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-gray-100 dark:border-gray-800">
                <span className="text-gray-500 dark:text-gray-400">PyTorch Version</span>
                <span className="font-mono text-gray-800 dark:text-gray-200">
                  {health?.pytorch_version || '2.x'}
                </span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-gray-100 dark:border-gray-800">
                <span className="text-gray-500 dark:text-gray-400">Hardware Acceleration</span>
                <span className="font-semibold text-gray-800 dark:text-gray-200">
                  {health?.cuda_available ? `GPU: ${health.device_name}` : 'CPU Execution'}
                </span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-gray-100 dark:border-gray-800">
                <span className="text-gray-500 dark:text-gray-400">Leakage Protection</span>
                <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-medium">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Fitted on Train Split
                </span>
              </div>
              <div className="flex items-center justify-between py-1.5">
                <span className="text-gray-500 dark:text-gray-400">Database Engine</span>
                <span className="font-mono text-gray-800 dark:text-gray-200">SQLite + WAL</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
