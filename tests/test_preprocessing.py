import numpy as np
import pytest
from PIL import Image

from defect_detection.preprocessing import PreprocessConfig, load_image_bytes, preprocess_pil, softmax, to_model_mode

from .conftest import encode

CFG = PreprocessConfig(image_size=64)


def _gradient(mode="L"):
    arr = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
    img = Image.fromarray(arr, mode="L")
    return img if mode == "L" else img.convert(mode)


def test_output_shape_and_dtype():
    x = preprocess_pil(_gradient(), CFG)
    assert x.shape == (3, 64, 64) and x.dtype == np.float32
    assert np.allclose(x[0], (x[0] * 1))  # finite
    assert np.isfinite(x).all()


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P", "LA"])
def test_colour_modes_give_same_tensor_as_grayscale(mode):
    ref = preprocess_pil(_gradient("L"), CFG)
    got = preprocess_pil(_gradient(mode), CFG)
    assert np.abs(ref - got).max() < 0.05


def test_rgba_transparency_composited_on_white():
    img = Image.new("RGBA", (80, 80), (0, 0, 0, 0))  # fully transparent black
    assert np.asarray(to_model_mode(img)).min() == 255


def test_16bit_images_are_rescaled():
    arr = (np.tile(np.linspace(0, 65535, 100), (100, 1))).astype(np.uint16)
    img = Image.fromarray(arr)  # uint16 -> mode I;16
    out = np.asarray(to_model_mode(img))
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() >= 254


def test_exif_orientation_is_applied():
    img = Image.new("L", (100, 50), 0)
    img.paste(255, (0, 0, 50, 50))  # left half white
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    rotated = to_model_mode(Image.open(__import__("io").BytesIO(encode(img, "JPEG", exif=exif))))
    assert rotated.size == (50, 100)


def test_load_image_bytes_rejects_truncated_file():
    data = encode(Image.new("RGB", (200, 200), (120, 50, 30)), "JPEG")
    with pytest.raises((OSError, SyntaxError)):
        load_image_bytes(data[: len(data) // 3])


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1000.0, 0.0], [0.0, 0.0]]))
    assert np.allclose(p.sum(1), 1) and np.isfinite(p).all()


def test_train_and_eval_pipelines_share_preprocessing():
    """With augmentation switched off the training transform must equal serving preprocessing."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import random

    from defect_detection.data.transforms import TrainTransform

    aug = {"crop_scale_min": 1.0, "brightness": 0.0, "contrast": 0.0, "blur_p": 0.0, "noise_p": 0.0}
    t = TrainTransform(CFG, aug)
    img = Image.new("L", (64, 64), 0)
    img.paste(200, (16, 16, 48, 48))  # dihedral-symmetric, so random flips/rotations cannot change it
    random.seed(0)
    got = t(img).numpy()
    ref = preprocess_pil(img, CFG)
    assert np.abs(got - ref).max() < 1e-5
