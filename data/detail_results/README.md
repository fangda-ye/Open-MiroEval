# Detailed Evaluation Results

This directory contains per-task per-model intermediate evaluation scores from our evaluation pipelines. These results can be used to reproduce the tables and figures in the paper. The files contain only numerical scores — no raw LLM reasoning, justifications, or evaluation content.

## Directory Structure

```
detail_results/
├── report/
│   ├── text.json              # Report quality scores (tasks 1-70, text-only)
│   └── multimodal.json        # Report quality scores (tasks 71-100, multimodal)
├── factuality/
│   ├── text.json              # Factuality verdict counts (text-only)
│   └── multimodal.json        # Factuality verdict counts (multimodal)
└── process/
    ├── text.json              # Process evaluation scores (text-only)
    └── multimodal.json        # Process evaluation scores (multimodal)
```

## File Formats

### report/{text,multimodal}.json

Per-task per-model report quality dimension scores. All scores are on a **0-10 float** scale.

```json
{
  "<task_id>": {
    "<model_id>": {
      "coverage": 7.766,                  // Coverage dimension score
      "insight": 7.280,                   // Insight dimension score
      "instruction_following": 8.665,     // Instruction-following score
      "clarity": 7.437,                   // Clarity score
      "query_spec": 6.101,               // Weighted avg of task-specific dimensions
      "total_weighted_score": 6.974       // Weighted total across ALL dimensions
    }
  }
}
```

The `query_spec` field is the weighted average of task-specific (non-fixed) dimensions for that task. The `total_weighted_score` is the overall score across all dimensions using the task-specific weights.

### factuality/{text,multimodal}.json

Per-task per-model factuality verdict counts (integers):

```json
{
  "<task_id>": {
    "<model_id>": {
      "right": 52,      // Statements verified as correct
      "wrong": 3,       // Statements contradicted by evidence
      "conflict": 0,    // Contradictory evidence (typically 0 for text-only, >0 for multimodal)
      "unknown": 8       // Statements that could not be verified
    }
  }
}
```

**Key metric:** `Right Ratio = right / (right + wrong + unknown + conflict) × 100`

Note: For text-only tasks, `conflict` is typically 0. For multimodal tasks, `conflict` reflects contradictions between web sources and document/image content.

### process/{text,multimodal}.json

Per-task per-model process evaluation scores. Dimension scores are **1-10 integers**.

```json
{
  "<task_id>": {
    "<model_id>": {
      "intrinsic": {
        "search_breadth": 7,
        "analytical_depth": 6,
        "progressive_refinement": 8,
        "critical_thinking": 7,
        "efficiency": 6
      },
      "alignment": {
        "findings_to_report": 8,
        "report_to_process": 5,
        "contradiction": 7
      }
    }
  }
}
```

Some entries include an additional `_overall_from_per_task` field alongside the normal dimension scores, representing the overall score derived from per-task aggregation:

```json
{
  "<task_id>": {
    "<model_id>": {
      "intrinsic": { "search_breadth": 8, "analytical_depth": 7, ... },
      "alignment": { "findings_to_report": 7, "report_to_process": 5, ... },
      "_overall_from_per_task": 7.033
    }
  }
}
```

