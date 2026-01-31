# 🧠 DocuSenseAI v2: Privacy-First Local Reasoning Engine

![Status](https://img.shields.io/badge/Status-Active_Development-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Inference](https://img.shields.io/badge/Inference-Local_CPU%2FGPU-green)

**DocuSenseAI v2** is a strictly local, air-gapped reasoning system designed to analyze sensitive documents and local file systems without data ever leaving the user's machine.

Unlike standard RAG (Retrieval-Augmented Generation) wrappers, this project implements a custom **"Lazy Discovery" architecture** that allows users to query their local hard drive *without* pre-indexing the entire disk, ensuring maximum privacy and minimal resource footprint.

---

## 🚀 Key Differentiators

### 1. Privacy-by-Constraint ("The Glass Box")
* **Zero Telemetry:** The system runs fully offline. Network calls to external APIs are architecturally impossible in the inference layer.
* **Transparency:** Every answer provides citations (Source Chunks). If the model "reads" a file, the user sees it in the Transparency Panel.

### 2. The "Disk Scout" Architecture (Novelty)
Instead of building a massive vector database of the user's entire hard drive (invasive & slow), DocuSenseAI does it as and when needed:
1.  **Intent Parsing:** A lightweight LLM extracts specific target keywords from natural language queries.
2.  **Deterministic Filtering:** The system scans file metadata in allowed folders.
3.  **Lazy Loading:** Only relevant files are opened, tokenized, and injected into the context window at query time.

### 3. Hardware-Adaptive Engineering
* **Optimized for Consumer Hardware:** Originally built on Llama-3 (8B), the inference engine was refactored to run on **Phi-3 Mini (3.8B)**.
* **Result:** Capable of running complex reasoning tasks on machines with **<4GB RAM**, preventing OOM (Out of Memory) crashes common in local AI.

---

## 🔄 Why DocuSenseAI v2? (The Evolution)

**DocuSenseAI v1** proved that local RAG was *possible*. **DocuSenseAI v2** proves it is *practical*.

Most local AI tutorials build "Happy Path" demos—they work perfectly if you upload a pristine text file and have 32GB of RAM. They fail in the real world where data is messy and hardware is constrained.

**v2 was re-engineered from the ground up to solve the "Toy Project" limitations:**

| Feature | v1 (The Prototype) | v2 (The Production Engine) |
| :--- | :--- | :--- |
| **File Handling** | Crashed on multi-sheet Excels or merged cells. | **Robust Ingestion Pipeline:** Custom Schema Injection detects headers, sanitizes `NaN` values, and handles multi-tab spreadsheets gracefully. |
| **Memory** | OOM (Out of Memory) crashes on standard laptops. | **Resource-Adaptive:** Swapped heavy 8B models for optimized **Phi-3 (3.8B)**, enabling stable inference on <4GB RAM environments. |
| **Disk Access** | All-or-nothing indexing (slow & invasive). | **Lazy Discovery:** "Scout" architecture separates intent (logic) from retrieval (IO), accessing files only when cryptographically verified against the allowlist. |
| **Safety** | trusted the model to "be nice." | **Adversarial Guardrails:** System prompts are engineered to fail safely ("I don't know") rather than hallucinate, with strict unlearning of outside knowledge. |

---

## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestrator** | Python + Pydantic | Logic control, type safety, and permission guardrails. |
| **Inference Engine** | **Ollama** (Phi-3 / Llama 3.2) | Local LLM serving and context management. |
| **Vector Store** | **FAISS** (CPU) | Ephemeral short-term memory for uploaded documents. |
| **Ingestion** | Pandas, Docx2txt, LangChain | Custom parsing pipeline for structured data (Excel/CSV). |
| **Interface** | Streamlit | Reactive UI with real-time transparency logs. |

---

## ⚡ Feature Spotlight: Smart Data Ingestion

LLMs notoriously struggle with **Excel** and **CSV** files due to tokenization limits and lost headers. DocuSenseAI v2 implements a custom **Schema Injection Pipeline**:

* **Multi-Sheet Awareness:** Automatically iterates through all Excel sheets.
* **Header Anchoring:** Extracts `df.columns` and explicitly injects a "Data Card" (Schema) at the top of the context window.
* **Result:** The model can accurately identify column names and data relationships even in messy, merged-cell spreadsheets.

---

## 💻 Installation & Usage

### Prerequisites
1.  **Ollama** installed and running.
2.  **Python 3.10+**

### Setup
```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/DocuSenseAI.git](https://github.com/yourusername/DocuSenseAI.git)
cd DocuSenseAI

# 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Pull the Optimized Model
ollama pull phi3

# 5. Run the engine
streamlit run app.py
```

---

## 🚧Roadmap & Evaluation Status
This project is currently in further development and evaluation - **Raj Aryan**
