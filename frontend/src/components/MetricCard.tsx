import React from 'react';
import clsx from 'clsx';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  accentColor?: 'emerald' | 'blue' | 'purple' | 'amber' | 'rose';
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subValue,
  icon: Icon,
  trend,
  trendValue,
  accentColor = 'emerald',
  className,
}) => {
  const colorMap = {
    emerald: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800/40',
    blue: 'bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800/40',
    purple: 'bg-purple-50 dark:bg-purple-950/40 text-purple-600 dark:text-purple-400 border-purple-200 dark:border-purple-800/40',
    amber: 'bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-800/40',
    rose: 'bg-rose-50 dark:bg-rose-950/40 text-rose-600 dark:text-rose-400 border-rose-200 dark:border-rose-800/40',
  };

  return (
    <div
      className={clsx(
        'bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm transition-all duration-200 hover:border-gray-300 dark:hover:border-gray-700',
        className
      )}
    >
      <div className="flex items-start justify-between">
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          {label}
        </span>
        {Icon && (
          <div className={clsx('p-2.5 rounded-lg border', colorMap[accentColor])}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      <div className="mt-3 flex items-baseline gap-2">
        <div className="text-2xl font-bold text-gray-900 dark:text-gray-50">
          {value}
        </div>
        {subValue && (
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {subValue}
          </span>
        )}
      </div>

      {trendValue && (
        <div className="mt-2.5 flex items-center text-xs">
          <span
            className={clsx(
              'font-medium mr-1.5',
              trend === 'up' && 'text-emerald-600 dark:text-emerald-400',
              trend === 'down' && 'text-rose-600 dark:text-rose-400',
              trend === 'neutral' && 'text-gray-500 dark:text-gray-400'
            )}
          >
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '•'} {trendValue}
          </span>
          <span className="text-gray-400 dark:text-gray-500">vs benchmark</span>
        </div>
      )}
    </div>
  );
};
