import React from 'react';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';
import clsx from 'clsx';

export type ToastType = 'success' | 'error' | 'info';

export interface ToastMessage {
  id: string;
  type: ToastType;
  message: string;
  title?: string;
}

interface ToastProps {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
}

export const ToastContainer: React.FC<ToastProps> = ({ toasts, onDismiss }) => {
  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2.5 max-w-sm w-full pointer-events-none">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={clsx(
            'pointer-events-auto flex items-start gap-3 p-4 rounded-xl border shadow-lg transition-all duration-300 animate-in slide-in-from-bottom-3',
            toast.type === 'success' &&
              'bg-emerald-50 dark:bg-emerald-950/90 border-emerald-200 dark:border-emerald-800 text-emerald-900 dark:text-emerald-100',
            toast.type === 'error' &&
              'bg-rose-50 dark:bg-rose-950/90 border-rose-200 dark:border-rose-800 text-rose-900 dark:text-rose-100',
            toast.type === 'info' &&
              'bg-blue-50 dark:bg-blue-950/90 border-blue-200 dark:border-blue-800 text-blue-900 dark:text-blue-100'
          )}
        >
          {toast.type === 'success' && <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />}
          {toast.type === 'error' && <AlertCircle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />}
          {toast.type === 'info' && <Info className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />}

          <div className="flex-1 text-sm">
            {toast.title && <div className="font-semibold">{toast.title}</div>}
            <div className="text-xs opacity-90 mt-0.5">{toast.message}</div>
          </div>

          <button
            onClick={() => onDismiss(toast.id)}
            className="p-1 hover:bg-black/5 dark:hover:bg-white/10 rounded-md transition-colors"
          >
            <X className="w-4 h-4 opacity-70 hover:opacity-100" />
          </button>
        </div>
      ))}
    </div>
  );
};
