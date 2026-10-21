# Bias-detection-and-mitigation

> **Investigating and mitigating demographic and presentation-order biases in large language model resume screening.**

This repository contains the codebase, audit datasets, benchmark results, and an interactive full-stack web application demonstrating a **Neurosymbolic AI Bias Mitigation Framework** for automated candidate screening.

---

## 📌 Project Overview

When Large Language Models (LLMs) such as Llama 3.2 or Qwen 2.5 are used to screen job applicants simultaneously (e.g., presenting 8 resumes with candidate names in a single prompt), they exhibit two severe failure modes:

1. **Demographic Disparities**: Unconscious preference for specific demographic groups (e.g., favoring White Female names at up to 43.3% vs. White Male names at 13.4% under time-pressure framing, $p < 0.001$).
2. **Positional Primacy Bias**: Resumes presented in earlier slots (Slots 1–3) are shortlisted at disproportionately high rates (55%–65%), whereas candidates in later slots (Slot 7) drop to 11%–15%. This causes **Quality Inversions** where clearly inferior candidates are chosen over superior candidates simply because they appeared earlier.

### 🛡️ The Neurosymbolic Solution

Our neurosymbolic pipeline replaces unconstrained multi-candidate prompting with a **fact-extraction, quote-grounded, and invariant scoring** architecture:
- **Candidate Blindness & Identity Separation**: Resumes are evaluated strictly in isolation without names or display slots.
- **Dynamic Rubric Induction**: Objective scoring rubrics (criteria, point weights, experience caps) are automatically compiled from Job Descriptions.
- **Symbolic Verification & Anti-Hallucination Grounding**: Factual claims extracted by the LLM are checked with exact/normalized substring and bigram overlap verifiers against the candidate's actual resume text.
- **Mathematical Top-$K$ Ranking**: Candidates are ranked purely on grounded, verified scores—guaranteeing **0.0% position bias**, **0.0% demographic disparity** under counterbalancing, and **0.0% quality inversions**.

---

## 📊 Benchmark Results Across 6 Professions (960 Trials)

Evaluated across **6 diverse professions** (Software Engineer, Elementary School Teacher, VP of Operations, Warehouse Associate, Physician, Journeyman Electrician) across 960 controlled audit trials:

| Profession | Framing | Baseline Top-3 Overlap | Neurosymbolic Overlap | Baseline Quality Inversions | Neurosymbolic Inversions | Errors Driven by Slot | Neurosymbolic Demographic Bias |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Software Engineer** | Neutral | 70.0% | **100.0%** | 20.5% | **0.0%** | 77.8% | **0.0%** |
| **Software Engineer** | Pressure | 58.3% | **100.0%** | 41.4% | **0.0%** | 57.1% | **0.0%** |
| **Teacher** | Neutral | 56.7% | **100.0%** | 15.8% | **0.0%** | 72.4% | **0.0%** |
| **Teacher** | Pressure | 50.8% | **100.0%** | 27.6% | **0.0%** | 67.1% | **0.0%** |
| **VP of Operations** | Neutral | 48.8% | **100.0%** | 24.0% | **0.0%** | 89.7% | **0.0%** |
| **VP of Operations** | Pressure | 51.2% | **100.0%** | 20.4% | **0.0%** | 89.4% | **0.0%** |
| **Warehouse Associate** | Neutral | 56.7% | **100.0%** | 23.9% | **0.0%** | 78.0% | **0.0%** |
| **Warehouse Associate** | Pressure | 54.2% | **100.0%** | 25.4% | **0.0%** | 73.0% | **0.0%** |
| **Physician** | Neutral | 58.3% | **100.0%** | 15.5% | **0.0%** | 77.5% | **0.0%** |
| **Physician** | Pressure | 57.1% | **100.0%** | 14.6% | **0.0%** | 88.9% | **0.0%** |
| **Electrician** | Neutral | 46.2% | **100.0%** | 39.4% | **0.0%** | 73.7% | **0.0%** |
| **Electrician** | Pressure | 47.9% | **100.0%** | 33.3% | **0.0%** | 74.3% | **0.0%** |

---

## 🖥️ Interactive Web Application (`app.py`)

A full-stack Flask + Chart.js web interface provides interactive experimentation and visualization:

1. **Interactive Screening Lab (Tab 1)**:
   - Configure all 8 candidate slots with any name and resume.
   - Test randomized slot orders or reverse slot assignments.
   - Inspect the exact raw prompt sent to the model under Neutral and Pressure framing.
   - Execute the Neurosymbolic pipeline to see ranked top-3 winners, verified scores, and experience levels.
   - Open the **"View Trace"** inspector to audit the exact model prompt, raw JSON reply, and quote grounding proofs.
   - Run the baseline unmitigated LLM to directly observe baseline bias.

2. **Visualizations & Audit Charts (Tab 2)**:
   - Browse audit datasets for all 6 professions (`results_*.csv` vs. `ns_results_*.csv`).
   - View 4 high-level KPI cards: Demographic Disparity Gap, Quality Inversion Rate, Slot-Driven Mistake Rate, and Primacy Position Skew.
   - Interactive charts:
     - **Demographic Selection Share**: Group win rates vs. 25% fair share baseline.
     - **Positional Primacy Bias**: Slot win rates vs. 37.5% fair share line.
     - **Resume Merit Win Distribution**: Win distribution across resumes R1–R8.
     - **Mitigation Impact Comparison**: Baseline error vs. Neurosymbolic 0.0% error rate.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) with `llama3.2` and optionally `qwen2.5:7b`:
  ```bash
  ollama pull llama3.2
  ollama pull qwen2.5:7b
  ```

### Installation
```bash
git clone https://github.com/varshiniharikrishna/Bias-detection-and-mitigation.git
cd Bias-detection-and-mitigation
pip install flask
```

### Launch the Web App
```bash
python app.py
```
Open **`http://localhost:5005`** in your browser.

---

## 🛠️ CLI Tools & Commands

### 1. Dynamic Rubric Induction
Extract an objective scoring rubric from any role's Job Description:
```bash
python neurosymbolic.py compile-rubric --role software_engineer --model llama3.2
```

### 2. Score Resumes
Score all candidates for a profession using the neurosymbolic engine:
```bash
python neurosymbolic.py score --role software_engineer --model llama3.2 --temp 0
```

### 3. Compare Baseline Audit vs. Neurosymbolic Ground Truth
Analyze an existing audit CSV for bias, demographic gaps, and position-driven inversions:
```bash
python neurosymbolic.py compare --role software_engineer results_eq.csv
python neurosymbolic.py compare --role teacher results_teacher.csv
```

### 4. Run Multi-Role Benchmark
Reproduce the 960-trial audit benchmark across all 6 roles:
```bash
python neurosymbolic.py benchmark
```

### 5. Check Determinism & Stability
Verify 0% score variance across repeated runs:
```bash
python neurosymbolic.py stability --role software_engineer --runs 5
```

---

## 📁 Repository Structure

```
├── app.py                      # Flask web application backend
├── templates/
│   └── index.html              # Interactive frontend UI (Lab & Visualizations)
├── neurosymbolic.py            # Neurosymbolic fact-extraction & verification engine
├── audit_roles.py              # Multi-role definitions (JDs, resumes, rubrics)
├── audit3.py                   # Census 2020 name audit runner & prompt framings
├── audit2.py                   # Baseline audit runner (Wilson & Caliskan names)
├── audit_equal_spread.py       # Latin-square counterbalanced trial generator
├── perception_check.py         # LLM perception tests on candidate names
├── Base_Resumes/               # Candidate resume templates
├── codes/                      # Auxiliary pilot experiments and analysis scripts
├── results_*.csv               # Baseline unmitigated audit trial datasets
├── ns_results_*.csv            # Neurosymbolic mitigated audit trial datasets
├── ns_*.json                   # Ground-truth verified scores and top-3 rankings
├── rubric_*.json               # Dynamically induced scoring rubrics
└── .gitignore                  # Git exclusions (literature papers, caches, etc.)
```
>>>>>>> ec55618 (Implement neurosymbolic bias mitigation pipeline, audit benchmarks, and interactive web dashboard)
