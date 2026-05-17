from __future__ import annotations


MODEL_FEATURE_CONTRACT = {
    "ln_change_rate_gated": "confidence_gated_value",
    "density_level": "stable_value",
    "density_burst": "stable_value",
    "hold_occupancy": "stable_value",
    "chord_ratio": "confidence_gated_or_high_support",
    "jack_excess": "sparse_tail_with_confidence",
    "jack_streak_exposure": "support_sensitive",
    "hand_balance_signed": "confidence_gated_value",
    "hand_imbalance_abs": "confidence_gated_value",
    "repeat_exact": "support_sensitive",
    "repeat_shift": "support_sensitive",
    "repeat_motion": "support_sensitive",
}

DIAGNOSTIC_FEATURE_CONTRACT = {
    "ln_change_rate_raw": "raw_value_with_side_confidence",
}

FEATURE_CONTRACT = {
    **DIAGNOSTIC_FEATURE_CONTRACT,
    **MODEL_FEATURE_CONTRACT,
}
