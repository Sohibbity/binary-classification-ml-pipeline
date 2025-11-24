# Binary Classification ML Pipeline

Production-grade ML pipeline leveraging a custom binary classifier (2-layer MLP) to predict customer subscription acquisition at scale.

## Overview

This project establishes robust ML infrastructure using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) as a proving ground. The architecture and patterns developed here lay the groundwork for a more ambitious goal: predicting short-term (4-6 week) price volatility in small-cap stocks to drive data-informed investment decisions.

**Note:** This project is under active development. Please disregard technical debt and comment clutter - cleanup and refactoring are ongoing.

## Tech Stack
- **ML Framework**: PyTorch
- **Cloud Infrastructure**: AWS S3, SageMaker
- **Data Processing**: Pandas, NumPy, boto3
- **Deployment**: SageMaker Endpoints with GPU acceleration

## Features

**Completed:**
- Custom PyTorch binary classifier with 2-layer MLP architecture
- S3-based streaming pipeline for memory-efficient batch processing
- Chunked inference with configurable batch sizes and retry logic
- Production-ready SageMaker inference contract (model_fn, input_fn, predict_fn, output_fn)


**In Progress:**
- SageMaker endpoint deployment with CUDA-enabled GPU inference

**Planned:**
- PostgreSQL integration for versioned prediction storage
- Real-time model performance monitoring and drift detection
- Airflow orchestration for scheduled batch jobs
- Migration to stock volatility prediction pipeline
- Unit tests - the lack thereof, they're a feature (until i break prod :))

## Architecture
```
S3 Raw Data → Streaming ETL → Model Inference (GPU) → S3 Predictions → PostgreSQL → Monitoring Dashboard
```

---

**Why This Matters**: Building production ML infrastructure with real constraints (streaming, GPU optimization, monitoring) on a simpler problem before tackling financial time-series prediction.