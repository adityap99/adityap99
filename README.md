# Hi there, I'm Aditya 👋

**Software Engineer (ML / AI Systems)** | **MS in Computer Science @ Georgia Tech** | **LLM & ML Infrastructure Enthusiast**

I’m currently pursuing my **MS in Computer Science at Georgia Tech** (GPA: 3.75/4.0, graduating May 2026), with a strong focus on **Machine Learning systems, LLM inference, and scalable data pipelines**.  
I bring **5+ years of industry experience** working on production Software Engineering and ML products, large-scale data systems, and performance-critical services at **Google X, Google Research**, and startups.

---

## 🚀 About Me

- 🎓 MS CS @ **Georgia Institute of Technology** (ML, DL, NLP, Systems for ML, Conversational AI)
- 🧠 ML Engineer with hands-on experience across **classical ML, deep learning, GenAI, and MLOps**
- ⚙️ Strong interest in **LLM inference systems, tail-latency mitigation, retrieval systems, and ML infrastructure**
- 🌍 Built and scaled ML systems used globally across **Search, climate risk modeling, and language learning**
- 👥 Enjoy mentoring, code quality improvements, and collaborating with domain experts (scientists & researchers)

---

## 💼 Professional Experience

### **Infocusp Innovations** — Software Engineer 2 | *Dec 2022 – Jun 2025*
**Client: Google X (Moonshot Factory)**

- Developed **multi-tenant hazard risk forecasting models** (1-, 2-, and 5-year) for heatwaves and wildfires across multiple regions; applied classical ML and deep learning algorithms on geospatial data, worked on the full pipeline from data ingestion (GEE, GCP) through modeling and explainability.
- Achieved **PR-AUC 0.96** in seismic activity classification and **Dice 0.73** in S-wave picking (segmentation) using DAS sensor data; introduced noise-robust training strategies for improved generalization.
- Accelerated model analysis pipelines, cutting experiment analysis execution time from **4 hours to 20 minutes**.
- Scaled geospatial data ingestion and processing using **Google Earth Engine, Pub/Sub, and GCP buckets**.
- Optimized data loading/batching to reduce class skew-induced drift from **24% to 1%**.
- Enhanced model interpretability with **SHAP, Grad-CAM, prediction histograms, and reliability curves**.
- Collaborated with climate scientists and implemented feature engineering for ML models. Mentored 3 junior engineers on best practices, improving repository code quality and increasing test coverage from **56% to 100%**.

**Client: Google Research**

- Owned, architected, and implemented product features and data pipelines for a **language learning platform** with **150K+ daily impressions** across India, LatAm, and Indonesia. Featured on the **Google Search results and Search Labs**.
- Diagnosed p99 latency regressions in a performance-sensitive API and refactored the critical path, achieving **~30% latency reduction**.

### **Infocusp Innovations** — Software Engineer 1 | *May 2020 – Dec 2022*
**Client: Google Research**

- Designed and implemented a notification cron-job for personalized communication across multiple channels. Introduced multi-threading and other optimizations to reduce processing time from **5 hours to 40 minutes**.
- Built scalable **gRPC microservices and API endpoints in C++** for the consumer app.

**Client: Bryte Labs**

- **Built an end-to-end pipeline** for biometric data ingestion, validation, visualization, cleanup, and modelling.

---
## 🧩 Work Portfolio

A selection of real-world products and platforms I’ve contributed to through industry roles and client engagements:

- **Google X – The Moonshot Factory**  
  Climate risk forecasting and geospatial ML systems  
  🌐 https://x.company/

- **Google Research**  
  Language learning platform featured on Google Search & Search Labs  
  🌐 https://research.google/

- **Bryte Labs**  
  Biometric data ingestion, modeling, and analytics platform  
  🌐 https://www.bryte.com/

- **Infocusp Innovations**  
  Engineering partner delivering ML and data systems for global clients  
  🌐 https://www.infocusp.com/

---

## 🧠 Academic & Research Projects

### 🔥 Risk-Triggered Mid-Flight Request Migration (LLM Inference)
- Built a **risk-triggered migration mechanism** for prefill–decode disaggregation in **vLLM**
- Mitigated decode stragglers under heavy-tailed workloads
- Achieved **65% TBT reduction** and **~25% lower end-to-end latency**
- Modeled KV-cache growth → time-between-tokens using telemetry + online calibration
- Implemented **stability-guarded migrations** (e.g., BF16 → FP8) on H100 GPUs

---

### 🔎 Hybrid Vector & Metadata Retrieval Search
- Designed a hybrid retrieval engine over **100K documents (768-D embeddings)**
- Implemented a **metadata-aware IVF index** for selective filtering
- Achieved **77× speedup** for high-selectivity queries (<1%)
- Benchmarked IVF-Flat, IVF-PQ, bitmap filtering, and brute-force baselines
- Identified ANN cross-over points across selectivity regimes

---

### 📄 Explainable RAG for SEC 10-K QA
- Built a **layout-aware, citation-grounded RAG pipeline**
- Implemented deterministic **faithfulness & verification checks**
- Designed retrieval and generation evaluation framework on **FinDER dataset**

---

### 🧩 Parameter-Efficient LLM Fine-Tuning
- Benchmarked **Full FT vs LoRA vs QLoRA** on a 7B LLM (50K NLI samples)
- LoRA within **~1%**, QLoRA within **~2%** of full fine-tuning (Acc / Macro-F1)
- Reduced peak GPU memory by **65–70%**
- Applied **context distillation** using KL-regularized ICL soft targets

---

### 🧠 Transformer Language Model from Scratch

- Implemented a **GPT-style Transformer language model** end-to-end in PyTorch, including embeddings, pre-norm Transformer blocks, RMSNorm, SwiGLU feed-forward layers, and causal multi-head self-attention with RoPE.
- Built the **full training pipeline** from scratch: memory-mapped data loading, batching, AdamW optimization, gradient clipping, checkpointing, and cosine learning rate scheduling.
- Trained a ~**17M parameter model** on the **TinyStories** dataset, achieving **≤ 1.8 validation loss**, with further improvements at higher token budgets.
- Implemented **autoregressive decoding** with temperature scaling and **top-p (nucleus) sampling** for text generation.
- (Optional) Implemented a **byte-level BPE tokenizer** with custom merge rules, encoding, and decoding.
- Passed a comprehensive **pytest-based test suite** covering attention, normalization, transformer blocks, training, and inference.

**Tech:** PyTorch, Transformers, RoPE, RMSNorm, SwiGLU, AdamW, CUDA, Pytest

> **Note:** Most academic project repositories are private due to Georgia Tech’s Office of Student Integrity policies.  
> I’m happy to discuss designs, trade-offs, and results—or provide demos for recruiting purposes.

---

## 🛠️ Tech Stack

**Languages:** Python • C++ • JavaScript • TypeScript • SQL  
**ML / AI:** PyTorch • TensorFlow • HuggingFace • vLLM • RAG • Vector Search • LoRA / QLoRA  
**MLOps & Observability:** MLflow • Weights & Biases • Prometheus • Grafana  
**Data & Infra:** GCP (GEE, Pub/Sub, BigQuery, Dataflow, Cloud Run) • AWS • Terraform  
**Systems:** Docker • Kubernetes • gRPC • CI/CD • Distributed Systems  

---

## 📚 Currently Exploring

- LLM inference optimization & scheduling
- Retrieval evaluation and faithfulness in RAG
- Systems for Machine Learning
- Tail-latency mitigation techniques

---

## 📫 Let’s Connect

- 💼 LinkedIn: https://linkedin.com/in/adityapandit99  
- 🧑‍💻 GitHub: https://github.com/adityap99  
- 📧 Email: adityaspandit99@gmail.com  

---

💡 **Open to roles in:**  
ML Engineering • AI Systems • LLM Infrastructure • Software Engineering (ML-heavy)

🎓 **Education:**  
MS Computer Science — Georgia Tech  
BTech ICT — DA-IICT
