import React, { useEffect, useRef } from 'react';
// @ts-ignore
import Plotly from 'plotly.js-dist-min';
import { useTheme } from '../context/ThemeContext';

interface PlotlyChartProps {
  data: any[];
  layout?: Record<string, any>;
  config?: Record<string, any>;
  style?: React.CSSProperties;
  className?: string;
}

export const PlotlyChart: React.FC<PlotlyChartProps> = ({
  data,
  layout = {},
  config = {},
  style,
  className,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  useEffect(() => {
    if (!containerRef.current) return;

    // Apply dark/light theme defaults
    const themedLayout = {
      paper_bgcolor: isDark ? '#111827' : '#ffffff',
      plot_bgcolor: isDark ? '#111827' : '#ffffff',
      font: {
        family: 'Inter, sans-serif',
        color: isDark ? '#F3F4F6' : '#111827',
        size: 12,
      },
      margin: { t: 40, r: 30, l: 50, b: 50, ...(layout.margin || {}) },
      autosize: true,
      ...layout,
    };

    const defaultConfig = {
      responsive: true,
      displayModeBar: false,
      ...config,
    };

    Plotly.newPlot(containerRef.current, data, themedLayout, defaultConfig);

    const handleResize = () => {
      if (containerRef.current) {
        Plotly.Plots.resize(containerRef.current);
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, [data, layout, config, isDark]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', minHeight: '320px', ...style }}
      className={className}
    />
  );
};
