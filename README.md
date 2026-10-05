# Out-of-Stock Retail Detection: Evaluating VLM Core Capabilities

This is a private repository dedicated to evaluating and highlighting the intrinsic limitations of frozen, non-fine-tuned Vision-Language Models (VLMs) when applied to a highly domain-specific task: **Out-of-Stock (OOS) retail shelf detection**.

---

## 🎯 Project Core Objective
The primary objective of this project is to demonstrate that state-of-the-art Multimodal Models, when used in a completely **zero-shot or prompt-engineered configuration (frozen parameters)**, struggle to guarantee robust reliability on specialized tasks like empty supermarket shelf detection. 

While the model shows solid generic grounding and visual capacities, it suffers from specific structural vulnerabilities—such as missing targets due to poor lighting, or generating false counts caused by deep shadows, camera motion blur, and refrigerator glass door reflections. This research establishes the critical necessity of domain-specific fine-tuning pipelines for industrial retail applications.

---

## 🤖 Target Evaluated Models & Scale Allocation
All experimental multi-modal pipelines within this repository are built around the **Qwen2.5-VL** architecture, evaluated across two distinct scales to assess the impact of parameter expansion on zero-shot domain adaptability:
* **`Qwen2.5-VL-3B-Instruct`**
* **`Qwen2.5-VL-7B-Instruct`**

### Model Deployment per Test Case:
* **Test 1 (Logit/Probability Mode):** Benchmarked using the **3B** model parameters to establish an initial performance floor.
* **Test 2 (Native Scaling - Binary Text):** Evaluated across both parameter sizes (**3B** vs **7B**) to cross-compare text-grounding alignment accuracy.
* **Test 3 (Multi-Modal Streams - 3-Level Logic):** Validated on both **3B** and **7B** structures to determine if parameter scaling reduces visual ambiguity.
* **Test 4 (Programmatic Reranking Pipeline):** Extensively executed on both model sizes (**3B** and **7B**) to quantify the effectiveness of rule-based recovery prompts on different model capacities.

---

## 📊 Dataset & Privacy Restrictions
> ⚠️ **Important Notice regarding Data Availability:** The evaluation datasets utilized to run the benchmarks consist of private, high-resolution surveillance streams capturing retail supermarket shelf gaps, utilized as part of the research conducted at the **MIVIA Lab (University of Salerno)**. Due to strict corporate non-disclosure agreements (NDAs) and privacy regulations, **the raw images and ground truth bounding box label annotations cannot be uploaded or distributed within this repository**.

However, the complete programmatic evaluation matrices, structural JSON evaluation outputs, and unbacked logging artifacts are maintained inside the localized project directory footprints.

---

## 📑 Experimental Framework & Test Specifications

The codebase evaluates the synergy between an initial object detector blueprint (**YOLO**) and sequential validation passes handled by the VLM (**Qwen2.5-VL**) through four evolutionary software stages.

### Test 1: Baseline Evaluation (Logit/Probability Mode)
* **Script:** `evaluate_yolo_qwen.py` (supported by `qwen_inference_module.py`).
* **Model Scale:** `Qwen2.5-VL-3B-Instruct`.
* **Objective:** Establish an initial assessment of the VLM's behavior by analyzing the underlying raw token probability distributions.
* **Input Structure:** The VLM receives the **full reference image** overlayed with annotated candidate bounding boxes (green for the target box under analysis, red for others).
* **Message / Prompt Protocol:** Executed via standard chat template structures. One contextual prompt query is sent per candidate box.
* **Output / Decision Mechanism:** The module isolates the logit scores for the individual `'yes'` and `'no'` text tokens directly from the model's final output position, applying a targeted binary softmax to calculate a soft probability distribution.
* **Main Targets Measured:** True Positive Rates (TPR), False Positive Rates (FPR), and False Negative Rates (FNR) cross-evaluated against the Ground Truth before and after filtering.

### Test 2: Native Scaling (Textual Binary Mode)
* **Script:** `evaluate_yolo_qwen_mm_v2.py` (supported by `qwen_inference_module_v2.py`).
* **Model Scales:** Comparative sweep between **3B** and **7B** parameters.
* **Objective:** Improve spatial grounding accuracy by transitioning from pixel coordinates to native system tokens and adopting a definitive text-driven decision mechanism.
* **Input Structure:** The full image is fed alongside the prompt. Bounding box pixel locations are transformed into Qwen's native internal integer scale (`[0, 1000]`).
* **Message / Prompt Protocol:** The template embeds structural text detection grounding tags: `<box>({x1},{y1}),({x2},{y2})</box>`. 
* **Output / Decision Mechanism:** Numerical thresholds are eliminated. Qwen directly outputs a definitive binary text token (`"yes"` or `"no"`). A confirmation bias directive enforces an affirmative fallback answer if the model generates uninterpretable strings.
* **Main Targets Measured:** Counts of raw VLM confirmations (`qwen_yes` vs `qwen_no`) evaluated against global YOLO predictions to track structural target preservation.

### Test 3: Multi-Modal Streams (Three-Level Tier Logic)
* **Script:** `evaluate_yolo_qwen_mm_v3.py` (supported by `qwen_inference_module_v3.py`).
* **Model Scales:** Verified on both **3B** and **7B** structures.
* **Objective:** Mitigate background ambiguity by feeding fine local close-up details and providing a safe margin for uncertainty.
* **Input Structure:** Dual multi-modal visual stream input. The model simultaneously processes the **full image** (for global illumination/context) and a **zoomed-in crop** of the target zone expanded with a 20% spatial padding margin.
* **Message / Prompt Protocol:** Prompts are expanded with a multi-step checklist. If an ambiguous zone triggers low confidence, a secondary pass is automatically initiated with an explanation prompt template.
* **Output / Decision Mechanism:** The decision layer expands to a three-level classification: `"yes"`, `"uncertain"`, or `"no"`. Ambiguous cases are counted separately, and their descriptive text justifications are saved into a dedicated log file (`uncertain_log.json`).
* **Main Targets Measured:** Density tracking of ambiguous shelf regions and granular compilation of qualitative failure explanations.

### Test 4: Programmatic Reranking Pipeline
* **Script:** `evaluate_yolo_qwen_mm_v4.py` (supported by `qwen_inference_module_v4.py`).
* **Model Scales:** Evaluated on **3B** and **7B** architectures to isolate scaling recovery margins.
* **Objective:** Correct lingering zero-shot errors by executing rule-based programmatic corrections and focused structural reranking on ambiguous data segments.
* **Input Structure:** Same as Test 3 (Dual-stream inputs).
* **Message / Prompt Protocol:** Integrates an advanced Phase 3 evaluation step. Ambiguous boxes are re-evaluated using a rules-based visual segmentation prompt:
  * Force `"YES"` if the region is dark, black, or covered by heavy shadows (lack of products).
  * Restrict `"NO"` answers only to cases where bright colors or clear geometric items are visible.
* **Output / Decision Mechanism:** Operates sequentially:
  1. Compiles manual blacklist overrides (`FORCED_NO_IDS`) to automatically discard known invalid streams.
  2. Executes a third verification forward pass for the remaining uncertain boxes.
* **Main Targets Measured:** Generates comparative pre-rerank (`metrics_v5.json`) and post-rerank (`metrics_v5_reranked.json`) statistical profiles to isolate recovery improvements.

---

## 📁 Repository Directory Structure

```text
Qwen Pipeline/
└── src/
    └── evaluation/
        ├── qwen_inference_module.py          # Logit-probability base interface
        ├── qwen_inference_module_V2.py       # [0,1000] Native coordinate text module
        ├── qwen_inference_module_v3.py       # 3-level tier logic + dual visual streams module
        ├── qwen_inference_module_v4.py       # Bound class utilities supporting reranking workflows
        ├── evaluate_yolo_qwen.py             # Test 1 evaluation routine
        ├── evaluate_yolo_qwen_mm_v2.py       # Test 2 evaluation routine
        ├── evaluate_yolo_qwen_mm_v3.py       # Test 3 evaluation routine
        ├── evaluate_yolo_qwen_mm_v4.py       # Test 4 evaluation routine
        ├── visualize_detector_only.py        # Baseline YOLO plot/table rendering suite
        ├── visualize_yolo_qwen.py            # Threshold-driven pipeline charting suite
        └── visualize_yolo_qwen_v2.py         # Modern cross-modal alignment table engine
