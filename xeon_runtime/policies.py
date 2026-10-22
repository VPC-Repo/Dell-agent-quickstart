"""Named prompt policies.

Grounding keeps generated text tied to supplied sources and tool results. It
is applied to every agent by default and disabled per agent with the body
config key ``grounding: false`` - a visible, versioned choice instead of an
unnamed suffix welded into every prompt.
"""

GROUNDING = (
    "Do not invent dates, authors, teams, or organizations. Omit an authorship "
    "line unless provided. Preserve supplied numeric values and units. Do not "
    "calculate new percentages or ratios in prose; use only comparisons "
    "explicitly computed by a tool, preserving the named numerator and "
    "denominator. Recommendations must be conditional on stated business "
    "priorities."
)
