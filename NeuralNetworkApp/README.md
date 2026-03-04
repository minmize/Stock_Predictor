# NeuroBuilder — Android Neural Network App

A beginner-friendly Android app for building, training, and understanding neural networks — entirely on-device.

## Features

### 📚 Interactive Tutorials (8 Chapters)
| Chapter | Topics |
|---------|--------|
| What is AI? | AI vs. ML, history, use cases |
| Neurons | Biological inspiration, weighted sum, activation |
| Layers | Input/Hidden/Output layers, sizing rules |
| Activation Functions | ReLU, Sigmoid, Tanh, Softmax, Linear |
| Training | Forward/backward pass, gradient descent |
| Loss Functions | MSE, Binary CE, Categorical CE |
| Overfitting | Underfitting, overfitting, dropout regularisation |
| Your First Model | Step-by-step guide |

### 🔨 Network Builder
- Type your architecture as a dash-separated string: `4-64-32-1`
- Visual layer cards — adjust neurons, activation, dropout per layer
- Drag to reorder hidden layers
- Live parameter count
- Pre-set architecture templates (Binary, Multi-class, Regression, Deep)
- Configurable: Learning rate, epochs, batch size, loss function

### 📂 Data Input & Processing
**File formats:** CSV, TSV, plain text (auto-detects separator)

**Sampling methods:**
- Use all data
- Random sample (N rows)
- First N / Last N rows
- Stratified sample

**Missing value strategies:**
- Drop rows
- Fill with mean / median / mode / zero

**Normalisation:**
- None
- Min-Max scaling (0–1)
- Z-score standardisation (mean=0, std=1)
- Robust scaling

**Outlier removal:**
- None
- IQR method (1.5× fence)
- Z-score method (|z| > 3)
- Percentile clipping (1st–99th)

**Other:**
- Configurable train/validation split (50%–95%)
- Column statistics (mean, median, mode, std, min, max, Q1, Q3)
- Column correlation matrix
- Built-in XOR sample dataset for testing

### 🏋️ Training
- Real-time loss curve chart (train + validation)
- Accuracy chart (classification tasks)
- Live epoch metrics
- Training log
- Stop/resume training
- SGD with momentum

### 📊 Results
- Final validation loss and accuracy
- Sample predictions vs. ground truth
- Manual prediction input (try any values)
- Full network architecture summary

---

## Architecture

```
app/src/main/java/com/neuralnet/builder/
├── engine/
│   ├── NeuralNetwork.kt       — Core NN: forward pass, backprop, SGD+momentum
│   └── ActivationFunctions.kt — ReLU, Sigmoid, Tanh, Softmax + loss functions
├── data/
│   ├── DataProcessor.kt       — CSV loading, sampling, normalisation, stats
│   └── DataInputActivity.kt   — UI: file picker, column selection, processing options
├── builder/
│   └── NetworkBuilderActivity.kt — UI: text-based & visual network builder
├── training/
│   └── TrainingActivity.kt    — UI: real-time loss/accuracy charts, training log
├── results/
│   └── ResultsActivity.kt     — UI: metrics, sample predictions, manual prediction
├── tutorial/
│   ├── TutorialActivity.kt    — ViewPager2-based tutorial viewer
│   └── TutorialData.kt        — All tutorial content
└── MainActivity.kt            — Home screen
```

## Neural Network Implementation

Pure Kotlin, no external ML libraries required. Implements:
- Dense (fully-connected) layers
- Multiple activation functions per layer
- He weight initialisation
- Mini-batch SGD with momentum (β=0.9)
- Numerically stable softmax (shifted by max)
- Sigmoid + BCE and Softmax + CCE simplified gradients

## Building

**Requirements:**
- Android Studio Hedgehog (2023.1.1) or later
- Android SDK 34
- Kotlin 1.9+

**Steps:**
1. Open `NeuralNetworkApp/` in Android Studio
2. Sync Gradle
3. Run on device or emulator (minSdk 26 = Android 8.0)

## Google Play Integration

This project is structured as a standard Android Gradle project.
To publish:
1. Create a signing keystore
2. Configure `signingConfigs` in `app/build.gradle`
3. Set `buildType release` to use signing config
4. Generate AAB: `./gradlew bundleRelease`
5. Upload to Google Play Console

## Performance Notes

- Training runs on a background coroutine (Dispatchers.Default)
- UI updates are posted to main thread every epoch
- Large datasets (>100k rows) should use sampling to avoid OOM
- `android:largeHeap="true"` is enabled in the manifest
