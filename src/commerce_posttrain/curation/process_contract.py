"""Actor-visible process feature contract shared by curation and evaluation.

This module freezes the observable fields and selection order before Teacher
collection.  The feature extractor implemented later must emit this schema and
must never populate evidence fields from hidden environment truth.
"""

PROCESS_CONTRACT_VERSION = "commerce-process-contract-v1"

ACTOR_VISIBLE_FEATURES = {
    "legality": (
        "guard_rejections",
        "malformed_calls",
        "schema_rejections",
    ),
    "search": (
        "unique_queries",
        "duplicate_queries",
        "visible_candidates",
    ),
    "candidate_use": (
        "opened_candidates",
        "compared_candidates",
    ),
    "evidence": (
        "required_detail_types_seen",
        "missing_required_detail_types",
        "option_axes_resolved",
        "final_price_visible",
    ),
    "termination": (
        "repeat_actions",
        "no_progress_actions",
        "steps_after_decision_ready",
    ),
    "context": (
        "projection_truncations",
        "max_input_tokens",
        "contract_violations",
    ),
    "diversity": (
        "tool_sequence_signature",
        "search_query_signature",
        "visited_product_signature",
        "first_divergence_turn",
    ),
}

# All numeric components are minimized.  Attempt index provides a stable final
# tie-break.  Hidden target fields are deliberately absent from this tuple.
PROCESS_SELECTION_ORDER = (
    "guard_rejections",
    "malformed_calls",
    "schema_rejections",
    "missing_required_detail_types",
    "repeat_actions",
    "no_progress_actions",
    "steps_after_decision_ready",
    "attempt_index",
)

EVIDENCE_SOURCE = "actor_visible_observation_only"
