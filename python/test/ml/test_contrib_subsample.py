import numpy as np
import pytest
from open3d.ml.contrib import subsample, subsample_batch


def assert_equal_2d_sort_by_row(x, y):
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2-dimensional.")
    x = np.sort(x, axis=0)
    y = np.sort(y, axis=0)
    np.testing.assert_equal(x, y)


def test_one():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                       [5, 0, 0], [5, 1, 0]],
                      dtype=np.float32)
    features = np.array(range(21), dtype=np.float32).reshape(-1, 3)
    labels = np.array(range(7), dtype=np.int32)

    # Reference results.
    sub_points_ref = np.array([[5, 0.5, 0], [0.4, 0.4, 0.4]], dtype=np.float32)

    # Passing only points.
    sub_points = subsample(points, sampleDl=1.1)
    assert_equal_2d_sort_by_row(sub_points, sub_points_ref)


def test_subsample():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                       [5, 0, 0], [5, 1, 0]],
                      dtype=np.float32)
    features = np.array(range(21), dtype=np.float32).reshape(-1, 3)
    labels = np.array(range(7), dtype=np.int32)

    # Reference results.
    sub_points_ref = np.array([[5, 0.5, 0], [0.4, 0.4, 0.4]], dtype=np.float32)
    sub_features_ref = np.array([[16.5, 17.5, 18.5], [6, 7, 8]],
                                dtype=np.float32)
    sub_labels_ref = np.array([6, 4], dtype=np.int32)

    # Passing only points.
    sub_points = subsample(points, sampleDl=1.1)
    np.testing.assert_equal(sub_points, sub_points_ref)

    # Passing points and features.
    sub_points, sub_features = subsample(points,
                                         features=features,
                                         sampleDl=1.1)
    np.testing.assert_equal(sub_points, sub_points_ref)
    np.testing.assert_equal(sub_features, sub_features_ref)

    # Passing points, features and labels.
    sub_points, sub_features, sub_labels = subsample(points,
                                                     features=features,
                                                     classes=labels,
                                                     sampleDl=1.1)
    np.testing.assert_equal(sub_points, sub_points_ref)
    np.testing.assert_equal(sub_features, sub_features_ref)
    np.testing.assert_equal(sub_labels, sub_labels_ref)

    # Test wrong dtype.
    with pytest.raises(RuntimeError):
        sub_points = subsample(np.array(points, dtype=np.int32), sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(np.array(points, dtype=np.float64), sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(points,
                               features=np.array(features, dtype=np.int32),
                               sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(points,
                               features=features,
                               classes=np.array(labels, dtype=np.float32),
                               sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(points,
                               features=np.array(features, np.int32),
                               classes=np.array(labels, dtype=np.float32),
                               sampleDl=1.1)

    # Test shape mismatch
    with pytest.raises(RuntimeError):
        sub_points = subsample(points[0], sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(np.ones((10, 4), dtype=np.float32), sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(points, features=features[0], sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(points[0], sampleDl=1.1)
    with pytest.raises(TypeError):
        sub_points = subsample(None, sampleDl=1.1)


def test_subsample_batch():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                       [5, 0, 0], [5, 1, 0]],
                      dtype=np.float32)
    features = np.array(range(28), dtype=np.float32).reshape(-1, 4)
    labels = np.array(range(7), dtype=np.int32)
    batches = np.array([3, 2, 2], dtype=np.int32)

    # Reference results.
    sub_points_ref = np.array(
        [[0.3333333, 0.3333333, 0], [0.5, 0.5, 1], [5, 0.5, 0]],
        dtype=np.float32)
    sub_batch_ref = np.array([1, 1, 1], dtype=np.int32)
    sub_labels_ref = np.array([2, 4, 6], dtype=np.int32)

    # Passing only points.
    sub_points, sub_batch = subsample_batch(points, batches, sampleDl=1.1)
    np.testing.assert_allclose(sub_points, sub_points_ref)
    np.testing.assert_allclose(sub_batch, sub_batch_ref)

    # Passing points and features.
    sub_features_ref = np.array(
        [[4, 5, 6, 7], [14, 15, 16, 17], [22, 23, 24, 25]], dtype=np.float32)
    sub_points, sub_batch, sub_features = subsample_batch(points,
                                                          batches,
                                                          features=features,
                                                          sampleDl=1.1)
    np.testing.assert_allclose(sub_points, sub_points_ref)
    np.testing.assert_allclose(sub_batch, sub_batch_ref)
    np.testing.assert_allclose(sub_features, sub_features_ref)

    # Passing points, features and labels.
    sub_points, sub_batch, sub_features, sub_labels = subsample_batch(
        points, batches, features=features, classes=labels, sampleDl=1.1)
    np.testing.assert_allclose(sub_points, sub_points_ref)
    np.testing.assert_allclose(sub_batch, sub_batch_ref)
    np.testing.assert_allclose(sub_features, sub_features_ref)
    np.testing.assert_equal(sub_labels, sub_labels_ref)

    # Test wrong dtype.
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(np.array(points, dtype=np.int32),
                                     batches,
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     np.array(batches, dtype=np.float32),
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(np.array(points, dtype=np.float64),
                                     batches,
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     batches,
                                     features=np.array(features,
                                                       dtype=np.int32),
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     batches,
                                     features=features,
                                     classes=np.array(labels, dtype=np.float32),
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     batches,
                                     features=np.array(features, np.int32),
                                     classes=np.array(labels, dtype=np.float32),
                                     sampleDl=1.1)

    # Test shape mismatch
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points[0], batches, sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(np.ones((10, 4), dtype=np.float32),
                                     batches,
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     batches,
                                     features=features[0],
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points[0], batches, sampleDl=1.1)
    with pytest.raises(TypeError):
        sub_points = subsample_batch(None, None, sampleDl=1.1)
    # Test sum(batch) != num_points
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     np.array([3, 3, 2], dtype=np.int32),
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     np.array([3, 3, 2], dtype=np.int32),
                                     features=features,
                                     classes=labels,
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     np.array([1], dtype=np.int32),
                                     sampleDl=1.1)
