import React from 'react';
import {
  LayoutDashboard,
  Database,
  Cpu,
  PlayCircle,
  GitCompare,
  BarChart3,
  History,
  Settings as SettingsIcon,
  Brain,
  ChevronLeft,
  ChevronRight,
  Layers,
  Wand2,
} from 'lucide-react';
import clsx from 'clsx';

interface SidebarProps {
  activePage: string;
  setActivePage: (page: string) => void;
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
  stats?: {
    total_datasets: number;
    total_experiments: number;
  };
}

export const Sidebar: React.FC<SidebarProps> = ({
  activePage,
  setActivePage,
  collapsed,
  setCollapsed,
  stats,
}) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'auto_train', label: 'Auto Clean & Train', icon: Wand2 },
    { id: 'datasets', label: 'Dataset Explorer', icon: Database, badge: stats?.total_datasets },
    { id: 'training', label: 'Model Training Lab', icon: Cpu },
    { id: 'prediction', label: 'Prediction Studio', icon: PlayCircle },
    { id: 'comparison', label: 'Model Comparison', icon: GitCompare },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'history', label: 'Experiment History', icon: History, badge: stats?.total_experiments },
    { id: 'settings', label: 'Settings', icon: SettingsIcon },
  ];

  return (
    <aside
      className={clsx(
        'h-screen sticky top-0 bg-white dark:bg-[#0B0F17] border-r border-gray-200 dark:border-gray-800 transition-all duration-300 z-40 flex flex-col justify-between shrink-0',
        collapsed ? 'w-20' : 'w-64'
      )}
    >
      <div>
        {/* Brand Header */}
        <div className="h-16 flex items-center justify-between px-5 border-b border-gray-100 dark:border-gray-800">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-emerald-700 flex items-center justify-center text-white shadow-md shadow-brand-500/20 shrink-0">
              <Brain className="w-5 h-5" />
            </div>
            {!collapsed && (
              <div className="leading-tight">
                <span className="font-bold text-base tracking-tight text-gray-900 dark:text-gray-100">
                  ML <span className="text-brand-500">STUDIO</span>
                </span>
                <span className="block text-[10px] text-gray-400 font-mono tracking-widest uppercase">
                  Workbench v1.0
                </span>
              </div>
            )}
          </div>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={clsx(
                  'w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all group relative',
                  isActive
                    ? 'bg-brand-50 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 font-semibold'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100/70 dark:hover:bg-gray-800/60 hover:text-gray-900 dark:hover:text-gray-200'
                )}
                title={collapsed ? item.label : undefined}
              >
                {isActive && (
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-brand-500 rounded-r-md" />
                )}
                <Icon
                  className={clsx(
                    'w-5 h-5 shrink-0 transition-transform group-hover:scale-105',
                    isActive ? 'text-brand-600 dark:text-brand-400' : 'text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300'
                  )}
                />
                {!collapsed && (
                  <span className="truncate flex-1 text-left">{item.label}</span>
                )}
                {!collapsed && item.badge !== undefined && item.badge > 0 && (
                  <span
                    className={clsx(
                      'px-2 py-0.5 rounded-full text-[11px] font-mono shrink-0',
                      isActive
                        ? 'bg-brand-100 dark:bg-brand-900/60 text-brand-800 dark:text-brand-200 font-bold'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400'
                    )}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info */}
      {!collapsed && (
        <div className="p-4 m-3 rounded-xl bg-gray-50 dark:bg-gray-900/50 border border-gray-100 dark:border-gray-800/80">
          <div className="flex items-center gap-2 text-xs font-semibold text-gray-800 dark:text-gray-200">
            <Layers className="w-4 h-4 text-brand-500" />
            <span>Dual Engine Ready</span>
          </div>
          <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-1">
            NumPy & PyTorch Scratch vs. Scikit-learn Reference
          </p>
        </div>
      )}
    </aside>
  );
};
