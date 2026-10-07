from app.liveness import LivenessChallenge


def feed(challenge, yaw, count=3):
    result = None
    for _ in range(count):
        result = challenge.observe(yaw, True)
    return result


def test_pose_sequence_completes():
    c = LivenessChallenge(steps=["center", "side_a", "side_b", "center"])
    feed(c, 0.0); feed(c, -.3); feed(c, .3); result = feed(c, 0.0)
    assert result["complete"]


def test_wrong_pose_does_not_advance():
    c = LivenessChallenge(steps=["side_a"])
    feed(c, .3, 10)
    assert c.index == 0


def test_missing_face_resets_hold():
    c = LivenessChallenge(steps=["center"])
    c.observe(0, True); c.observe(0, True); c.observe(None, False)
    assert c.hold_frames == 0
