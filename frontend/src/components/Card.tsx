import React from 'react';
import clsx from 'clsx';

interface CardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  children,
  action,
  className,
  bodyClassName,
}) => {
  return (
    <div
      className={clsx(
        'bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 rounded-xl shadow-sm overflow-hidden transition-all duration-200',
        className
      )}
    >
      {(title || subtitle || action) && (
        <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-800/80 flex items-center justify-between flex-wrap gap-2">
          <div>
            {title && (
              <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                {subtitle}
              </p>
            )}
          </div>
          {action && <div className="flex items-center gap-2">{action}</div>}
        </div>
      )}
      <div className={clsx('p-6', bodyClassName)}>{children}</div>
    </div>
  );
};
