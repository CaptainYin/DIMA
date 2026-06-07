from __future__ import annotations


MPE_BENCHMARK_SCENARIOS = (
    "simple_spread",
    "simple_push",
    "simple_tag",
    "simple_adversary",
)

MPE_SCENARIO_ALIASES = {
    "simple_spread": "simple_spread_v2",
    "simple_spread_v2": "simple_spread_v2",
    "simple_push": "simple_push_v2",
    "simple_push_v2": "simple_push_v2",
    "simple_tag": "simple_tag_v2",
    "simple_tag_v2": "simple_tag_v2",
    "simple_adversary": "simple_adversary_v2",
    "simple_adversary_v2": "simple_adversary_v2",
}


def normalize_mpe_scenario(scenario: str) -> str:
    key = scenario.strip()
    if key not in MPE_SCENARIO_ALIASES:
        supported = ", ".join(sorted(MPE_SCENARIO_ALIASES))
        raise ValueError(f"Unsupported MPE scenario '{scenario}'. Supported: {supported}")
    return MPE_SCENARIO_ALIASES[key]
