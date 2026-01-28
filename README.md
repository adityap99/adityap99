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

### **Infocusp Innovations** — Software Engineer 2  
**Client: Google X (Moonshot Factory)** | *Jan 2022 – Jun 2025*

- Built **multi-tenant hazard risk forecasting models** (1–5 year horizons) for heatwaves and wildfires using geospatial data
- Achieved **PR-AUC 0.96** (seismic classification) and **Dice 0.73** (S-wave segmentation) on DAS sensor data
- Reduced experiment analysis runtime from **4 hours → 20 minutes**
- Scaled ingestion pipelines using **Google Earth Engine, Pub/Sub, and GCP buckets**
- Reduced class-skew-induced model drift from **24% → 1%**
- Added explainability with **SHAP, Grad-CAM, reliability curves**
- Mentored junior engineers; increased test coverage from **56% → 100%**

---

### **Infocusp Innovations** — Software Engineer  
**Client: Google Research** | *May 2020 – Dec 2021*

- Owned product features and data pipelines for a **language learning platform** with **150K+ daily impressions**
- Platform featured on **Google Search results and Search Labs**
- Designed notification cron jobs; reduced runtime from **5 hours → 40 minutes**
- Built **gRPC microservices in C++**
- Diagnosed and fixed p99 latency regressions, achieving **~30% latency reduction**

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
