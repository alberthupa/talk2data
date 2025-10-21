Below are three practical “cascade” pipelines that start simple/fast and only escalate to heavier methods when confidence/thresholds are not met. Each pipeline uses your techniques_basic.md primitives, defines concrete thresholds and margins, and covers continuation vs switch vs none. You can pick one, or combine ideas.

Common setup (applies to all pipelines)
- Data assets
  - Topic catalog: id, name, short_desc, synonyms, 1–3 positive examples, 1–2 near-miss negatives.
  - Precompute topic embeddings; optionally topic centroids for clusters.
- Output schema (strict)
  - { decision: "continue" | "switch" | "none", topic_id: string | null, confidence: 0..1, method: string, evidence: {scores, top_candidates, thresholds, stage} }
- Multi-turn logic
  - Maintain prev_topic and last_n turns (n=1–3). Compute separate similarity for continuation vs switch.
- Calibration
  - Use a small validation set to set thresholds for: accept (high precision), margin (top1−top2), and abstain. Target high precision on “accept” (≥0.9), allow 15–30% to escalate.

Recommended default models
- Embeddings: E5-small/large, bge-small-en, text-embedding-3-large
- Cross-encoder: ms-marco-MiniLM-L-6-v2 (fast) or a DeBERTa NLI
- LLM (cheap): gpt-4o-mini / Claude Haiku / Llama 3.1 8B
- LLM (heavier, only on hard cases): gpt-4o / Claude Sonnet

Pipeline A: Embedding-first, LLM tie-break (fastest typical)
- Goal: Resolve most turns via a single embedding search; only borderlines go to a cheap LLM reasoner.
- Steps
  1) Continuation quick check
     - Compute sim(user, prev_topic). If sim ≥ cont_threshold (e.g., 0.72) AND (sim − next_best_sim) ≥ cont_margin (e.g., 0.05) → decision = continue.
  2) Dense vector top-k
     - ANN search (k=5–10). If top1_sim ≥ accept_threshold (e.g., 0.70) AND (top1 − top2) ≥ margin (e.g., 0.07) → decision = switch to top1.
     - If top1_sim ≤ none_threshold (e.g., 0.45) → decision = none.
  3) Cheap LLM tie-break over candidates
     - Provide user text, brief history, and top-k topic snippets (name + desc + 1 short example).
     - Ask model to pick topic_id or return none/continue; require model_conf ≥ llm_accept (e.g., 0.60) and if switch, also require embedding top1_sim ≥ llm_floor (e.g., 0.55) to avoid hallucinated classes.
     - Else → escalate (optional) to cross-encoder OR return none (depending on your abstain policy).
- Pros: Lowest latency for majority; minimal infra. 
- Cons: Borderline semantics rely on LLM tie-break; quality of topic descriptions matters.

Pipeline B: Cheap lexical prefilter → small LLM → embeddings as guardrail → cross-encoder only if needed
- Goal: Keep tokens low in LLM by first narrowing with BM25; use embeddings and a cross-encoder only on tough cases.
- Steps
  1) Continuation quick check (same as A).
  2) BM25 prefilter
     - Index topic docs (name + desc + synonyms). Retrieve top_m (e.g., m=20).
     - If no hits with decent BM25 score → immediately go to embeddings (Step 4) or none if very weak.
  3) Cheap LLM over top_m
     - Few-shot JSON classification among those m candidates + option none/continue.
     - If model_conf ≥ 0.70 and margin_to_second ≥ 0.10 → accept.
  4) Embedding sanity check
     - If chosen topic exists, verify embedding sim(user, chosen_topic) ≥ 0.55; else treat as low-confidence → Step 5.
     - If no chosen topic (LLM abstained), run embedding top-k; if top1 ≥ 0.70 and margin ≥ 0.07 → accept; if ≤ 0.45 → none; else Step 5.
  5) Cross-encoder finalize (only for unresolved)
     - Pairwise score for top-k (from BM25 ∪ dense). If ce_top ≥ 0.65 and (ce_top − ce_2nd) ≥ 0.08 → accept; else none.
- Pros: Very cheap most of the time; good for large catalogs due to lexical shrink, conservative via embedding guardrail.
- Cons: Requires BM25 infra; two-stage logic slightly more plumbing.

Pipeline C: Two-level semantic router → cross-encoder → RAG LLM only for hardest cases
- Goal: Structural routing to keep candidate sets tiny, then a precise but still cheap scorer; LLM reasoning is last resort.
- Steps
  1) Continuation quick check (same as A).
  2) Coarse semantic routing
     - Cluster topics offline (e.g., k-means); store cluster centroids.
     - Assign message to cluster via centroid similarity. If cluster_top_sim < coarse_floor (e.g., 0.50) → none.
  3) In-cluster embedding retrieval
     - Top-k within cluster. If top1 ≥ 0.72 and margin ≥ 0.08 → accept; if ≤ 0.45 → none; else Step 4.
  4) Cross-encoder rerank
     - If ce_top ≥ 0.65 and margin ≥ 0.08 → accept; else Step 5.
  5) RAG LLM over top-3
     - Provide short topic docs for top-3; ask LLM to decide switch/continue/none with rationale; require llm_accept ≥ 0.65. If still ambiguous → none.
- Pros: Scales well as topics grow; stable decisions via cluster gating; cross-encoder yields strong precision.
- Cons: Offline clustering adds ops; slightly higher complexity.

Continuation vs switch decision policy (for all pipelines)
- Compute both:
  - sim_prev = sim(user, prev_topic)
  - sim_top1 = sim(user, best_new_topic)
- Rules:
  - If sim_prev ≥ cont_threshold AND (sim_prev − sim_top1) ≥ delta_threshold → continue.
  - Else if best_new_topic chosen by the stage logic → switch.
  - Else none.
- Hysteresis to prevent flip-flop:
  - Use slightly higher threshold to switch away from a prev_topic than to continue it (e.g., +0.03 to +0.05).
  - Require stability: two consecutive turns below cont_threshold before switching on marginal cases.

Concrete defaults to start (tune on validation later)
- cont_threshold: 0.72
- cont_margin/delta_threshold: 0.05
- accept_threshold (embedding): 0.70
- none_threshold (embedding): 0.45
- margin (embedding top1−top2): 0.07–0.10
- ce_accept: 0.65; ce_margin: 0.08
- llm_accept (cheap LLM): 0.60–0.70 depending on model
- k candidates: 5–10; BM25 m: 20; RAG LLM last resort top_k: 3

Pseudocode skeleton for a cascade
```
function route_turn(user, history, prev_topic):
  cand = {}
  # Step 1: continuation quick check
  if prev_topic:
    sim_prev = embed_sim(user, prev_topic)
    top_alt = top_alt_sim(user, exclude=prev_topic)  # from ANN
    if sim_prev >= 0.72 and (sim_prev - top_alt) >= 0.05:
      return {decision: "continue", topic_id: prev_topic, confidence: sim_prev, method: "cont_sim"}

  # Step 2: fast path (embedding)
  topk = ann_search(user, k=10)
  if topk[0].sim >= 0.70 and (topk[0].sim - topk[1].sim) >= 0.07:
    return {decision: "switch", topic_id: topk[0].id, confidence: topk[0].sim, method: "dense_fast"}

  if topk[0].sim <= 0.45:
    return {decision: "none", topic_id: null, confidence: 1 - topk[0].sim, method: "dense_floor"}

  # Step 3: tie-break (choose one per pipeline)
  # A: cheap LLM over top-k
  llm_out = llm_classify(user, history_last2, candidates=topk[:5])
  if llm_out.conf >= 0.60 and (get_sim(user, llm_out.topic_id) >= 0.55):
    return {decision: llm_out.decision, topic_id: llm_out.topic_id, confidence: llm_out.conf, method: "llm_tiebreak"}

  # Optional: cross-encoder
  ce = cross_encoder_rerank(user, topk[:5])
  if ce.top.score >= 0.65 and (ce.top.score - ce.second.score) >= 0.08:
    return {decision: "switch", topic_id: ce.top.id, confidence: ce.top.score, method: "cross_encoder"}

  # Final abstain
  return {decision: "none", topic_id: null, confidence: 0.0, method: "abstain"}
```

Operational notes
- Logging/telemetry: Record scores per stage, chosen thresholds, top-2 deltas; this is critical for calibration.
- Topic growth: Append-only updates are easy for embeddings; periodically re-cluster (C) or refresh BM25 (B).
- Guardrails: When LLM picks a topic absent in top-k, require a minimal embedding sim floor to accept.
- Token efficiency: Keep candidate snippets to ~50–80 tokens each; cap top_k passed to LLM at 3–5.
- Evaluation: Calibrate thresholds to hit your desired precision/recall tradeoff and acceptable escalation rate.

If you want, I can draft a minimal router.yaml (thresholds + stages) and a Python skeleton that implements Pipeline A or B. Toggle to Act mode when you’re ready for code.