import numpy as np
from defense.tracking.tracker import VideoTracker
from defense.localization.fusion_engine import FusionEngine

def test_tracker_proximity_gating_and_merging():
    """VideoTracker should never create duplicate tracks for a single drone in the same area."""
    fusion = FusionEngine(mode="adaptive")
    config = {
        "max_distance_threshold": 100,
        "min_track_confidence": 2,
        "track_memory": 10
    }
    tracker = VideoTracker(detector=None, localizer=None, fusion_engine=fusion, config=config)

    # Frame 1: Single detection at (500, 500)
    tracker.fusion.fuse = lambda *args, **kwargs: [
        {"center_x": 500, "center_y": 500, "width": 40, "height": 30, "confidence": 0.85, "polygon": None}
    ]
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker.process_frame(frame)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 1

    # Frame 2: Association succeeds, but suppose an unmatched spurious detection appears 15px away
    tracker.fusion.fuse = lambda *args, **kwargs: [
        {"center_x": 505, "center_y": 505, "width": 40, "height": 30, "confidence": 0.90, "polygon": None},
        {"center_x": 515, "center_y": 510, "width": 45, "height": 35, "confidence": 0.70, "polygon": None}
    ]
    tracker.process_frame(frame)
    # The second detection at (515, 510) is within proximity gating distance of track 1
    # It must NOT create a new track!
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 1

def test_tracker_colocated_track_merging():
    """If two tracks somehow become co-located, they must be merged into a single track."""
    fusion = FusionEngine(mode="adaptive")
    config = {"max_distance_threshold": 100, "min_track_confidence": 2, "track_memory": 10}
    tracker = VideoTracker(detector=None, localizer=None, fusion_engine=fusion, config=config)

    # Manually insert two tracks in the same area
    from defense.tracking.drone_track import DroneTrack
    t1 = DroneTrack(track_id=1, bbox=(500, 500, 40, 30), confidence=0.8, min_confirm_frames=2)
    t1.confirmed = True
    t1.frames_tracked = 5

    t2 = DroneTrack(track_id=2, bbox=(510, 505, 42, 32), confidence=0.7, min_confirm_frames=2)
    t2.confirmed = False
    t2.frames_tracked = 1

    tracker.tracks = [t1, t2]
    tracker._merge_colocated_tracks()

    # Should be merged into track 1
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 1
    assert tracker.tracks[0].confirmed
