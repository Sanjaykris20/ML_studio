import { Sun, Moon, Activity, Cpu, Sparkles, Database, Upload } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { SystemHealth } from '../types';

interface NavbarProps {
  systemHealth?: SystemHealth | null;
  activePage: string;
  onUploadClick?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  systemHealth,
  activePage,
  onUploadClick,
}) => {
  const { theme, toggleTheme } = useTheme();

  const getPageTitle = (page: string) => {
    switch (page) {
      case 'dashboard':
        return 'Overview Dashboard';
      case 'auto_train':
        return 'Auto Clean & Train Pipeline';
      case 'datasets':
        return 'Dataset Explorer & Ingestion';
      case 'training':
        return 'Model Training Workbench';
      case 'prediction':
        return 'Prediction Studio';
      case 'comparison':
        return 'Model Comparison Lab (Scratch vs. Library)';
      case 'analytics':
        return 'Model Analytics & Visualizations';
      case 'history':
        return 'Experiment Tracking History';
      case 'settings':
        return 'Platform Settings & Diagnostics';
      default:
        return 'ML Studio';
    }
  };

  return (
    <header className="h-16 border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-[#0B0F17]/80 backdrop-blur-md sticky top-0 z-30 px-6 flex items-center justify-between transition-colors">
      <div className="flex items-center gap-4">
        <div>
          <h1 className="text-lg font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <span>{getPageTitle(activePage)}</span>
          </h1>
          <p className="text-xs text-gray-500 dark:text-gray-400 hidden sm:block">
            Full-Stack Machine Learning Workbench & Benchmark Lab
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Backend Connectivity Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border bg-gray-50 dark:bg-gray-900/60 border-gray-200 dark:border-gray-800">
          <span
            className={`w-2 h-2 rounded-full ${
              systemHealth?.status === 'healthy' ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
            }`}
          />
          <span className="text-gray-600 dark:text-gray-300 hidden md:inline">
            {systemHealth?.status === 'healthy' ? 'Engine Online' : 'Connecting Engine...'}
          </span>
          {systemHealth?.cuda_available && (
            <span className="bg-emerald-100 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300 px-1.5 py-0.5 rounded text-[10px] uppercase font-bold">
              CUDA
            </span>
          )}
        </div>

        {/* Quick Upload CSV Action */}
        {onUploadClick && (
          <button
            onClick={onUploadClick}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-xs font-semibold shadow-xs transition-all cursor-pointer"
            title="Upload a new CSV dataset"
          >
            <Upload className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Upload CSV</span>
          </button>
        )}

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          aria-label="Toggle theme"
          className="p-2 rounded-lg border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 transition-colors"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-blue-500" />}
        </button>
      </div>
    </header>
  );
};
