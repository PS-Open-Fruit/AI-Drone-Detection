from defense.tracking.drone_track import DroneTrack

def test_track_confirmation():
    """Track should be confirmed after min_confirm_frames updates."""
    track = DroneTrack(track_id=1, bbox=(0, 0, 10, 10), min_confirm_frames=3)
    assert not track.confirmed
    track.update((1, 1, 10, 10), 0.9)
    track.update((2, 2, 10, 10), 0.9)
    assert track.confirmed  # 3 frames total (1 init + 2 updates)

def test_track_death():
    """Track should die after max_lost_frames of no detections."""
    track = DroneTrack(track_id=1, bbox=(0, 0, 10, 10), max_lost_frames=5)
    for _ in range(6):
        track.mark_lost()
    assert track.is_dead

def test_track_color_cycling():
    """Track colors should cycle through the palette."""
    t1 = DroneTrack(track_id=0, bbox=(0, 0, 10, 10))
    t2 = DroneTrack(track_id=10, bbox=(0, 0, 10, 10))
    assert t1.color == t2.color  # both index 0 in palette
