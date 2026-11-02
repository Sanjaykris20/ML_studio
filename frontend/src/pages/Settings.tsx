import React, { useState, useEffect } from 'react';
import {
  Settings as SettingsIcon,
  Sun,
  Moon,
  Database,
  Cpu,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  HardDrive,
  Sliders,
} from 'lucide-react';
import { Card } from '../components/Card';
import { useTheme } from '../context/ThemeContext';
import { SystemHealth } from '../types';
import { getSystemHealth } from '../services/api';

interface SettingsProps {
  showToast: (type: 'success' | 'error' | 'info', message: string, title?: string) => void;
}

export const Settings: React.FC<SettingsProps> = ({ showToast }) => {
  const { theme, toggleTheme, setTheme } = useTheme();
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  const fetchHealth = async () => {
    try {
      setIsRefreshing(true);
      const data = await getSystemHealth();
      setHealth(data);
      showToast('success', 'System diagnostics refreshed.');
    } catch (err) {
      showToast('error', 'Failed to reach ML backend service.');
    } finally {
      setIsRefreshing(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="space-y-6 max-w-4xl animate-in fade-in duration-300">
      {/* Page Header */}
      <div className="bg-white dark:bg-[#111827] p-5 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm flex items-center justify-between">
        <div className="flex items-center gap-3">
          <SettingsIcon className="w-5 h-5 text-brand-500 shrink-0" />
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-gray-100">
              Platform Settings & Diagnostics
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Interface preferences, self-configuring storage, and runtime specifications
            </p>
          </div>
        </div>

        <button
          onClick={fetchHealth}
          disabled={isRefreshing}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 text-xs font-semibold text-gray-700 dark:text-gray-300 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span>Refresh Status</span>
        </button>
      </div>

      {/* Theme Preferences */}
      <Card
        title="Appearance & Theme"
        subtitle="Custom SaaS color system with smooth day and night modes"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <button
            onClick={() => setTheme('dark')}
            className={`p-4 rounded-xl border text-left transition-all flex items-start gap-3 ${
              theme === 'dark'
                ? 'border-brand-500 bg-brand-50/50 dark:bg-brand-950/40 ring-2 ring-brand-500/30'
                : 'border-gray-200 dark:border-gray-800 hover:border-gray-300 dark:hover:border-gray-700'
            }`}
          >
            <div className="p-2.5 rounded-lg bg-gray-900 text-amber-400 shrink-0">
              <Moon className="w-5 h-5" />
            </div>
            <div>
              <div className="text-sm font-bold text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
                <span>Dark Theme</span>
                {theme === 'dark' && <CheckCircle2 className="w-3.5 h-3.5 text-brand-500" />}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Optimized for code analysis and extended training monitoring.
              </p>
            </div>
          </button>

          <button
            onClick={() => setTheme('light')}
            className={`p-4 rounded-xl border text-left transition-all flex items-start gap-3 ${
              theme === 'light'
                ? 'border-brand-500 bg-brand-50/50 dark:bg-brand-950/40 ring-2 ring-brand-500/30'
                : 'border-gray-200 dark:border-gray-800 hover:border-gray-300 dark:hover:border-gray-700'
            }`}
          >
            <div className="p-2.5 rounded-lg bg-gray-100 text-blue-500 shrink-0">
              <Sun className="w-5 h-5" />
            </div>
            <div>
              <div className="text-sm font-bold text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
                <span>Light Theme</span>
                {theme === 'light' && <CheckCircle2 className="w-3.5 h-3.5 text-brand-500" />}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                High contrast clean presentation theme suitable for daylight environments.
              </p>
            </div>
          </button>
        </div>
      </Card>

      {/* Backend & Runtime Diagnostics */}
      <Card
        title="Runtime & Environment Specifications"
        subtitle="Self-configuring zero-setup local database and machine learning engines"
      >
        <div className="divide-y divide-gray-100 dark:divide-gray-800 text-xs">
          <div className="py-3 flex items-center justify-between">
            <span className="text-gray-500 dark:text-gray-400">Backend API Connectivity</span>
            <span className="flex items-center gap-1.5 font-semibold text-emerald-600 dark:text-emerald-400">
              <CheckCircle2 className="w-4 h-4" />
              {health?.status === 'healthy' ? 'FastAPI Operational' : 'Degraded / Reconnecting'}
            </span>
          </div>

          <div className="py-3 flex items-center justify-between">
            <span className="text-gray-500 dark:text-gray-400">Python Runtime</span>
            <span className="font-mono font-medium text-gray-900 dark:text-gray-100">
              v{health?.python_version || '3.x'}
            </span>
          </div>

          <div className="py-3 flex items-center justify-between">
            <span className="text-gray-500 dark:text-gray-400">PyTorch Acceleration Engine</span>
            <span className="font-mono font-medium text-gray-900 dark:text-gray-100">
              {health?.pytorch_version || '2.x'} ({health?.cuda_available ? 'CUDA GPU Enabled' : 'CPU Mode'})
            </span>
          </div>

          <div className="py-3 flex items-center justify-between">
            <span className="text-gray-500 dark:text-gray-400">Active Device Name</span>
            <span className="font-semibold text-gray-900 dark:text-gray-100">
              {health?.device_name || 'CPU'}
            </span>
          </div>

          <div className="py-3 flex items-center justify-between">
            <span className="text-gray-500 dark:text-gray-400">Local SQLite Database File</span>
            <span className="font-mono text-[11px] text-gray-700 dark:text-gray-300 truncate max-w-sm">
              {health?.database_path || 'ml_studio.db'}
            </span>
          </div>

          <div className="py-3 flex items-center justify-between">
            <span className="text-gray-500 dark:text-gray-400">Artifacts Persistence Mode</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              Local Filesystem (backend/artifacts/models)
            </span>
          </div>
        </div>
      </Card>
    </div>
  );
};
