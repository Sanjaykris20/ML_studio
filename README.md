# ML Studio – Interactive Machine Learning Workbench

A full-stack, interactive machine learning workbench that allows users to upload datasets, preprocess data with strict data leakage prevention, train machine learning algorithms implemented **both from scratch (NumPy & PyTorch autograd)** and **via established reference libraries (Scikit-learn)**, evaluate performance, compare parity side-by-side, generate real-time predictions, and track experiment histories.

---

## 1. System Architecture

```
                                  +---------------------------------------+
                                  |     React 18 + TypeScript + Vite      |
                                  | Tailwind CSS • Recharts • Plotly.js   |
                                  +-------------------+-------------------+
                                                      |
                                           REST API (JSON / FormData)
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         FastAPI REST Backend          |
                                  |  Pydantic Schemas • SQLite Lifecycle  |
                                  +---------+-------------------+---------+
                                            |                   |
                     +----------------------+                   +-----------------------+
                     v                                                                  v
+------------------------------------------+                       +------------------------------------------+
|       Data Preprocessing Pipeline        |                       |          Dual Execution Engines          |
|  * Train-only fitting (Zero leakage)     |                       |  1. Scratch (NumPy & PyTorch autograd)   |
|  * Missing value imputation              |                       |  2. Library (Scikit-learn reference)     |
|  * One-Hot / Ordinal categorical encode  |                       +--------------------+---------------------+
|  * StandardScaler / MinMaxScaler         |                                            |
+------------------------------------------+                                            v
                                                                   +------------------------------------------+
                                                                   |         Evaluation Metrics Suite         |
                                                                   |  Regression: R², MAE, MSE, RMSE, Plots   |
                                                                   |  Classification: Acc, F1, ROC-AUC, CM    |
                                                                   +--------------------+---------------------+
                                                                                        |
                                                                                        v
                                                                   +------------------------------------------+
                                                                   |         Persistence Layer                |
                                                                   |  * SQLite: Experiments, Datasets, Logs   |
                                                                   |  * Disk: Serialized Pipeline Bundles     |
                                                                   +------------------------------------------+
```

---

## 2. Directory Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI application entrypoint & lifespan seeder
│   │   ├── core/
│   │   │   └── config.py             # Self-configuring environment & storage paths
│   │   ├── db/
│   │   │   ├── session.py            # SQLite engine, foreign keys, & built-in datasets seeder
│   │   │   └── models.py             # SQLAlchemy ORM models (Datasets, Experiments, Predictions)
│   │   ├── schemas/                  # Pydantic schemas for data validation
│   │   │   ├── dataset.py
│   │   │   ├── training.py
│   │   │   ├── prediction.py
│   │   │   └── experiment.py
│   │   ├── ml/
│   │   │   ├── preprocessing.py      # Preprocessor with zero data leakage guarantees
│   │   │   ├── evaluation.py         # Diagnostic metrics suite (R², RMSE, ROC, Confusion Matrix)
│   │   │   ├── pipeline.py           # Self-contained pipeline bundle persistence manager
│   │   │   ├── scratch/
│   │   │   │   ├── linear_regression.py   # NumPy Linear Regression + MSE + Gradient Descent
│   │   │   │   ├── logistic_regression.py # NumPy Logistic Regression + BCE + Sigmoid
│   │   │   │   └── mlp.py                 # PyTorch autograd MLP (Regression & Classification)
│   │   │   └── library/
│   │   │       └── sklearn_models.py      # Scikit-learn Linear, Logistic, and MLP wrappers
│   │   └── api/
│   │       ├── health.py             # Diagnostic health checks & stats
│   │       ├── datasets.py           # Upload, preview, & correlation matrix
│   │       ├── models.py             # Training lab, prediction studio, & comparison
│   │       └── experiments.py        # Experiment tracking & artifact management
│   ├── tests/                        # Automated Pytest suite (16 tests)
│   ├── artifacts/
│   │   ├── datasets/                 # Seeded & user-uploaded CSV files
│   │   └── models/                   # Serialized pipeline bundles (.joblib)
│   ├── ml_studio.db                  # Local SQLite database (created automatically)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/               # Navbar, Sidebar, Card, MetricCard, PlotlyChart, Toast
│   │   ├── context/                  # ThemeContext (Persistent Dark/Light mode)
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx         # Overview metrics, stats, & recent runs
│   │   │   ├── DatasetExplorer.tsx   # Upload, inspect, stats, & correlation heatmaps
│   │   │   ├── TrainingLab.tsx       # Hyperparameters, live loss curves, & test metrics
│   │   │   ├── PredictionStudio.tsx  # Dynamic form inputs, confidence meters, & history
│   │   │   ├── ModelComparison.tsx   # Side-by-side Scratch vs Library delta benchmark
│   │   │   ├── Analytics.tsx         # Deep visual diagnostics & ROC curves
│   │   │   ├── ExperimentHistory.tsx # Filterable history, inspect modal, & delete
│   │   │   └── Settings.tsx          # Appearance, system diagnostics, & storage specs
│   │   ├── services/                 # Axios REST client
│   │   ├── types/                    # TypeScript interfaces
│   │   ├── App.tsx                   # Main SPA shell & notification provider
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── tsconfig.json
├── start_backend.bat                 # Windows startup script for FastAPI
├── start_frontend.bat                # Windows startup script for Vite
├── start_project.bat                 # Windows dual launcher for complete platform
├── Dockerfile.backend                # Container specification for backend
├── Dockerfile.frontend               # Production Nginx multi-stage build
├── docker-compose.yml                # Multi-container orchestration
├── .env.example
├── .gitignore
└── README.md
```

---

## 3. Mathematical Foundations & Algorithm Explanations

### A. Linear Regression (NumPy From Scratch)
1. **Hypothesis / Prediction Function**:
   $$\hat{y} = X w + b$$
   where $X \in \mathbb{R}^{m \times d}$, $w \in \mathbb{R}^d$, and $b \in \mathbb{R}$.

2. **Mean Squared Error (MSE) Cost Function**:
   $$J(w, b) = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{m} \|Xw + b\mathbf{1} - y\|_2^2$$

3. **Analytical Gradients**:
   $$\frac{\partial J}{\partial w} = \frac{2}{m} X^T (\hat{y} - y)$$
   $$\frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$

4. **Gradient Descent Updates**:
   $$w \leftarrow w - \alpha \frac{\partial J}{\partial w}, \quad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
   where $\alpha$ is the configurable learning rate.

---

### B. Logistic Regression (NumPy From Scratch)
1. **Linear Combination**:
   $$z = X w + b$$

2. **Sigmoid Activation Function**:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
   *Numerical Stability*: $z$ is clipped into $[-250, 250]$ to prevent floating-point overflow in $e^{-z}$.

3. **Probability & Binary Cross-Entropy (BCE) Loss**:
   $$p = \sigma(z)$$
   $$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(p^{(i)} + \epsilon) + (1 - y^{(i)}) \ln(1 - p^{(i)} + \epsilon) \right]$$
   where $\epsilon = 10^{-15}$ avoids $\ln(0)$.

4. **Analytical Gradients**:
   $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (p - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (p^{(i)} - y^{(i)})$$

5. **Decision Threshold**:
   $$\hat{y} = \begin{cases} 1 & \text{if } p \ge \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

---

### C. Multilayer Perceptron (PyTorch Autograd)
1. **Forward Propagation**:
   For layer $l = 1, \dots, L$:
   $$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \quad a^{[l]} = g^{[l]}(z^{[l]})$$
   where $g^{[l]}$ is the selected activation function ($\text{ReLU}(z) = \max(0, z)$, $\text{Sigmoid}$, or $\text{Tanh}$).

2. **Backpropagation**:
   Computed using PyTorch computational graphs and reverse-mode automatic differentiation (`loss.backward()`).

3. **Optimizers**:
   - **Adam**: Computes adaptive learning rates with exponentially decaying averages of past gradients ($m_t$) and squared gradients ($v_t$):
     $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
   - **SGD with Momentum**:
     $$v_t = \gamma v_{t-1} + \alpha \nabla_\theta J(\theta), \quad \theta \leftarrow \theta - v_t$$

4. **Multi-Task Loss Functions**:
   - Regression: $\text{MSELoss}$
   - Binary Classification: $\text{CrossEntropyLoss}$ ($C = 2$)
   - Multiclass Classification: $\text{CrossEntropyLoss}$ ($C > 2$ classes)

---

## 4. Built-In Benchmark Datasets

The platform automatically seeds 3 standardized benchmark datasets from Scikit-learn on initial launch:
1. **Diabetes Progression**: 442 rows, 10 continuous features (age, sex, bmi, bp, blood serum measurements). Ideal for Linear Regression and MLP Regression.
2. **Breast Cancer Wisconsin**: 569 rows, 30 continuous clinical features. Ideal for Logistic Regression binary classification.
3. **Optical Digits 8x8**: 1,797 rows, 64 pixel features representing digits 0 through 9. Ideal for Multilayer Perceptron multiclass classification.

---

## 5. Strict Data Leakage Prevention

To ensure valid benchmarks and prevent optimistic performance inflation:
1. The dataset is partitioned into $X_{\text{train}}$ and $X_{\text{test}}$ **before** any transformations are computed.
2. All scalers (`StandardScaler`, `MinMaxScaler`), imputers (`mean`, `median`, `most_frequent`), and encoders (`OneHotEncoder`, `OrdinalEncoder`) are fit strictly on $X_{\text{train}}$.
3. The exact fitted transformer instances are serialized into the `PipelineBundle`. When new predictions are generated in the Prediction Studio, inputs pass through the saved fitted transformations without re-computing statistics.

---

## 6. Windows Local Startup Instructions

### Quick Launch (Both Frontend & Backend)
Double-click:
```cmd
start_project.bat
```
This automatically starts:
- **Backend API**: `http://127.0.0.1:8000` (Swagger docs at `http://127.0.0.1:8000/docs`)
- **Frontend Dashboard**: `http://localhost:5173`

### Manual Startup

#### Step 1: Start Backend
```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Step 2: Start Frontend
```powershell
cd frontend
npm install
npm run dev
```

---

## 7. Running Automated Tests

Run the full automated test suite (verifying math correctness, gradient finite differences, convergence, and API endpoints):

```powershell
$env:PYTHONPATH = "backend"
python -m pytest backend/tests -v
```

All 16 test suites run in under 10 seconds:
- `test_linear_regression_cost_calculation` (PASSED)
- `test_linear_regression_numerical_gradient_check` (PASSED)
- `test_linear_regression_convergence_and_loss_decrease` (PASSED)
- `test_sigmoid_numerical_stability` (PASSED)
- `test_bce_cost_calculation` (PASSED)
- `test_logistic_regression_numerical_gradient_check` (PASSED)
- `test_logistic_regression_convergence` (PASSED)
- `test_mlp_regression` (PASSED)
- `test_mlp_binary_classification` (PASSED)
- `test_mlp_multiclass_classification` (PASSED)
- `test_preprocessing_leakage_prevention` (PASSED)
- `test_preprocessing_missing_value_imputation` (PASSED)
- `test_health_endpoint` (PASSED)
- `test_datasets_list_and_preview` (PASSED)
- `test_train_scratch_linear_regression_and_predict` (PASSED)
- `test_train_library_linear_regression_and_compare` (PASSED)

---

## 8. Containerized Cloud Deployment

The project includes production-ready Docker configurations:

```bash
docker-compose up --build -d
```

- Backend runs in Python 3.11-slim container on port `8000`.
- Frontend is compiled into static assets and served via lightweight Nginx on port `80` with automatic reverse proxy routing to `/api`.
- Model artifacts and SQLite database are persisted inside the `ml_studio_data` Docker volume.
