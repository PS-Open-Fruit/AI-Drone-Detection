from defense.localization.feature_encoder import encode_features

def test_feature_vector_length():
    """Feature encoder should return a vector of exactly 20 elements."""
    box = (100, 200, 50, 30)
    shape = (720, 1280)
    features = encode_features(box, shape)
    assert len(features) == 20

def test_center_box_normalized_coords():
    """A box at the exact image center should have nx ≈ 0, ny ≈ 0."""
    box = (615, 335, 50, 50)  # center of 1280x720 is (640, 360), center of box is (640, 360)
    shape = (720, 1280)
    features = encode_features(box, shape)
    assert abs(features[0]) < 0.01  # nx ≈ 0
    assert abs(features[1]) < 0.01  # ny ≈ 0

def test_area_ratio_range():
    """Area ratio should be between 0 and 1."""
    box = (0, 0, 100, 100)
    shape = (1080, 1920)
    features = encode_features(box, shape)
    assert 0 < features[4] < 1
