import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { ToastContainer, ToastMessage, ToastType } from './components/Toast';
import { Dashboard } from './pages/Dashboard';
import { DatasetExplorer } from './pages/DatasetExplorer';
import { TrainingLab } from './pages/TrainingLab';
import { PredictionStudio } from './pages/PredictionStudio';
import { ModelComparison } from './pages/ModelComparison';
import { Analytics } from './pages/Analytics';
import { ExperimentHistory } from './pages/ExperimentHistory';
import { Settings } from './pages/Settings';
import { AutoCleanTrain } from './pages/AutoCleanTrain';
import { SystemHealth, DashboardStats } from './types';
import { getSystemHealth, getDashboardStats } from './services/api';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<string>('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  // Inter-page navigation transfer state
  const [navDatasetId, setNavDatasetId] = useState<string | undefined>(undefined);
  const [navModelId, setNavModelId] = useState<string | undefined>(undefined);
  const [openUploadModal, setOpenUploadModal] = useState<boolean>(false);

  const showToast = (type: ToastType, message: string, title?: string) => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, type, message, title }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4500);
  };

  const dismissToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const handleOpenUpload = () => {
    setOpenUploadModal(true);
    setActivePage('datasets');
  };

  const refreshSystemData = async () => {
    try {
      const [h, s] = await Promise.all([getSystemHealth(), getDashboardStats()]);
      setSystemHealth(h);
      setStats(s);
    } catch (err) {
      console.warn('System status check failed:', err);
    }
  };

  useEffect(() => {
    refreshSystemData();
    const interval = setInterval(refreshSystemData, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectDatasetForTraining = (datasetId: string) => {
    setNavDatasetId(datasetId);
    setActivePage('training');
  };

  const handleNavigateToPrediction = (experimentId: string) => {
    setNavModelId(experimentId);
    setActivePage('prediction');
  };

  return (
    <div className="flex min-h-screen bg-gray-50 dark:bg-[#0B0F17] text-gray-900 dark:text-gray-100 font-sans transition-colors duration-200">
      {/* Sidebar Navigation */}
      <Sidebar
        activePage={activePage}
        setActivePage={(page) => {
          setActivePage(page);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }}
        collapsed={sidebarCollapsed}
        setCollapsed={setSidebarCollapsed}
        stats={
          stats
            ? {
                total_datasets: stats.total_datasets,
                total_experiments: stats.total_experiments,
              }
            : undefined
        }
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar
          systemHealth={systemHealth}
          activePage={activePage}
          onUploadClick={handleOpenUpload}
        />

        <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto">
          {activePage === 'dashboard' && (
            <Dashboard
              setActivePage={setActivePage}
              onUploadClick={handleOpenUpload}
            />
          )}

          {activePage === 'auto_train' && (
            <AutoCleanTrain
              showToast={showToast}
              onNavigateToPrediction={handleNavigateToPrediction}
            />
          )}

          {activePage === 'datasets' && (
            <DatasetExplorer
              onSelectDatasetForTraining={handleSelectDatasetForTraining}
              initialOpenUpload={openUploadModal}
              onCloseUpload={() => setOpenUploadModal(false)}
              showToast={showToast}
            />
          )}

          {activePage === 'training' && (
            <TrainingLab
              initialDatasetId={navDatasetId}
              onNavigateToPrediction={handleNavigateToPrediction}
              onNavigateToUpload={handleOpenUpload}
              showToast={showToast}
            />
          )}

          {activePage === 'prediction' && (
            <PredictionStudio
              initialModelId={navModelId}
              showToast={showToast}
            />
          )}

          {activePage === 'comparison' && <ModelComparison showToast={showToast} />}

          {activePage === 'analytics' && <Analytics showToast={showToast} />}

          {activePage === 'history' && (
            <ExperimentHistory
              onNavigateToPrediction={handleNavigateToPrediction}
              showToast={showToast}
            />
          )}

          {activePage === 'settings' && <Settings showToast={showToast} />}
        </main>
      </div>

      {/* Notification Toasts */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
};
export default App;
