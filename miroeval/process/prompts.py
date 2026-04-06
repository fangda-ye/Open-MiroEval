INTRINSIC_EVAL_PROMPT = """\
You are an expert evaluator assessing the quality of a deep research agent's investigation process.

## Input

**User Query:**
{query}

**Structured Process:**
{structured_process}

## Task

Evaluate the research process on the following 5 dimensions. Score each from 1-10 with a brief justification.

### Dimensions

1. **search_breadth** (1-10): Did the agent explore diverse information sources and angles? Did it search for multiple aspects of the query rather than just one narrow perspective?
   - 1-3: Minimal searching, only 1-2 queries on the same narrow topic
   - 4-6: Some diversity but missed important angles
   - 7-10: Comprehensive search covering multiple relevant dimensions

2. **analytical_depth** (1-10): Did the agent analyze information deeply rather than just collecting surface-level facts? Did it interpret, compare, and draw insights from the gathered data?
   - 1-3: Merely listed facts without analysis
   - 4-6: Some analysis but mostly surface-level
   - 7-10: Deep analysis with meaningful insights and interpretations

3. **progressive_refinement** (1-10): Did the agent iteratively deepen its understanding? Did later steps build on earlier findings? Did the research trajectory show increasing sophistication?
   - 1-3: Flat trajectory, no visible learning across steps
   - 4-6: Some progression but largely parallel/independent steps
   - 7-10: Clear iterative deepening with each stage building on previous discoveries

4. **critical_thinking** (1-10): Did the agent question assumptions, cross-check information, identify contradictions, or acknowledge limitations?
   - 1-3: No evidence of critical evaluation
   - 4-6: Occasional questioning but mostly accepted information at face value
   - 7-10: Actively verified claims, noted discrepancies, and evaluated source reliability

5. **efficiency** (1-10): Was the process efficient? Did each step contribute meaningfully, or were there redundant searches, circular reasoning, or wasted effort?
   - 1-3: Highly redundant, many wasted steps
   - 4-6: Some redundancy but generally productive
   - 7-10: Lean process where almost every step contributed value

## Output Format

Return a JSON object:
```json
{{
  "search_breadth": {{"score": 8, "justification": "..."}},
  "analytical_depth": {{"score": 7, "justification": "..."}},
  "progressive_refinement": {{"score": 6, "justification": "..."}},
  "critical_thinking": {{"score": 5, "justification": "..."}},
  "efficiency": {{"score": 7, "justification": "..."}}
}}
```
"""


ALIGNMENT_EVAL_PROMPT = """\
You are an expert evaluator assessing the alignment between a research agent's investigation process and its final report.

## Input

**User Query:**
{query}

**Key Findings from Process:**
{global_findings}

**Final Report (may be truncated):**
{report}

## Task

Evaluate the alignment between the process findings and the final report on 3 dimensions. Score each from 1-10 with a brief justification.

### Dimensions

1. **findings_to_report** (1-10): What fraction of the key findings discovered during the process actually appear in the final report?
   - 1-3: Most process findings are absent from the report
   - 4-6: Some findings appear but many important ones are missing
   - 7-10: Nearly all process findings are incorporated into the report

2. **report_to_process** (1-10): Can the major claims and conclusions in the report be traced back to findings in the process? Or does the report contain substantial content that has no basis in the documented process?
   - 1-3: Report contains many claims with no process basis (possible hallucination)
   - 4-6: Most report claims have some process basis but some appear unsupported
   - 7-10: Nearly all report content is traceable to process findings

3. **contradiction** (1-10): Are the process findings and report conclusions consistent with each other? (10 = perfectly consistent, no contradictions)
   - 1-3: Major contradictions between process and report
   - 4-6: Minor inconsistencies or selective presentation
   - 7-10: Process and report are fully consistent

## Output Format

Return a JSON object:
```json
{{
  "findings_to_report": {{"score": 8, "justification": "..."}},
  "report_to_process": {{"score": 7, "justification": "..."}},
  "contradiction": {{"score": 9, "justification": "..."}}
}}
```
"""


STRUCTURING_PROMPT = """\
You are an expert analyst tasked with converting a deep research agent's raw process trace into a structured representation.

## Input

**User Query:**
{query}

**Raw Process Trace:**
{process_text}

## Task

Analyze this process trace and extract a structured representation. The trace may come from different AI research agents and can be in various formats (step-by-step logs, thinking traces, search results, narrative descriptions, etc.).

For each identifiable step in the process, determine:
1. **action_type**: one of: `plan`, `search`, `read`, `analyze`, `synthesize`, `verify`
   - `plan`: Setting research strategy, decomposing the task, identifying what to investigate
   - `search`: Issuing search queries, browsing for information sources
   - `read`: Reading/scraping specific documents, extracting information from sources
   - `analyze`: Deep analysis of gathered information, reasoning about findings
   - `synthesize`: Combining information from multiple sources, drafting conclusions
   - `verify`: Cross-checking facts, validating claims, identifying contradictions
2. **summary**: A concise 1-2 sentence description of what this step accomplished
3. **key_findings**: Specific factual claims or insights that were **newly discovered in this step**. Do NOT repeat findings already listed in a previous step. If a step merely confirms or reuses earlier information, leave key_findings empty or only note genuinely new information. Failed attempts, dead ends, and resource constraints encountered in this step should also be recorded here.

After analyzing all steps, identify the **global_findings**: the most important conclusions that emerge from synthesizing information **across multiple steps**.

## Output Format

Return a JSON object with exactly this structure:
```json
{{
  "steps": [
    {{
      "step_id": 1,
      "action_type": "plan",
      "summary": "...",
      "key_findings": ["...", "..."]
    }}
  ],
  "global_findings": [
    {{
      "finding": "A cross-step conclusion or synthesized insight",
      "first_found_at_step": 3,
      "related_steps": [1, 5, 8],
      "evidence_strength": "strong"
    }}
  ]
}}
```

For `evidence_strength`, use: `strong`, `moderate`, or `weak`.

## Guidelines
- Merge very small consecutive steps of the same type into one step
- Do NOT invent findings not present in the trace
- Keep global_findings focused on the most substantive discoveries (aim for 5-15 findings)
- Pay special attention to failed searches, abandoned paths, and strategy changes.
"""


COMPRESS_PROMPT = """\
You are a research process trace compressor. Compress the raw trace while preserving all information needed to evaluate research quality.

**User Query (for context):**
{query}

**This is chunk {chunk_idx} of {total_chunks} from the full process trace.**

**Raw Process Trace Chunk:**
{chunk_text}

## PRESERVE: search queries, reasoning/analysis, strategy decisions, key findings, failed attempts, cross-checking, chronological order.
## REMOVE: full web page text, duplicates, raw HTML/JSON, formatting tags, verbose tool calls, boilerplate snippets.

Output compressed trace as concise prose, preserving chronological flow. Do NOT add commentary.
"""
