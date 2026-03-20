"""
gui.complication_manager -- Perimeter layout engine for complications.

Responsible for:
  - Positioning complications around the circular perimeter ring
  - Managing focus/blur transitions when an overlay opens
  - Delegating draw calls to each complication
  - Hit-testing taps against complication positions

Will be fully implemented in migration step 2.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from gui.complications.base import BaseComplication


class ComplicationManager:
    """Lays out complications on the perimeter and composites overlays."""

    def __init__(self, bus) -> None:  # noqa: ANN001
        self.bus = bus
        self.focus_comp: Optional[str] = None
        self._focus_anim: float = 0.0

    def layout(
        self,
        complications: List[BaseComplication],
        cx: float,
        cy: float,
        mind: float,
    ) -> List[Tuple[BaseComplication, float, float, float]]:
        """Compute (comp, x, y, rotation_deg) for each complication."""
        n = max(1, len(complications))
        margin_frac = 0.038
        base = mind * 0.085
        comp_size = base * 2.35
        perim_margin = mind * margin_frac
        outer_margin = perim_margin * 0.35
        max_radius = comp_size * 0.5
        rim_r = (mind * 0.5) - perim_margin - outer_margin - max_radius

        positions = []
        for i, comp in enumerate(complications):
            theta = -math.pi / 2 + i * (2 * math.pi / n)
            x = cx + rim_r * math.cos(theta)
            y = cy + rim_r * math.sin(theta)
            rot = math.degrees(theta) + 90
            positions.append((comp, x, y, rot))
        return positions
