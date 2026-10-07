import numpy as np
from app.compositor import SemanticCompositor


def test_expanded_bbox_clamps_to_frame():
    box = SemanticCompositor._expanded_bbox((-10, -10, 120, 120), 100, 80)
    assert box == (0, 0, 100, 80)


def test_class_sets_protect_glasses_and_hair():
    from app.compositor import SKIN, PROTECTED_OCCLUDERS
    assert 6 in PROTECTED_OCCLUDERS
    assert 17 in PROTECTED_OCCLUDERS
    assert not (SKIN & PROTECTED_OCCLUDERS)
