"""The command-line entries import without the STL engine and keep their defaults."""

from stlia import train_denoise, train_patches


def test_train_denoise_defaults():
    a = train_denoise.parser().parse_args(["--data", "D", "--out", "O"])
    assert (a.nside, a.noise, a.realizations, a.steps, a.lr, a.real) == (64, "pr3sims", 4, 3000, 1e-3, False)
    assert a.loss == "data=1,res=1,cross=1"


def test_train_denoise_real_mode():
    a = train_denoise.parser().parse_args(["--data", "D", "--out", "O", "--real", "--loss", "data=1,l1=1"])
    assert a.real and a.loss == "data=1,l1=1"


def test_train_patches_defaults():
    a = train_patches.parser().parse_args(["--data", "D", "--out", "O"])
    assert (a.nside, a.j, a.radius, a.core, a.patches, a.stl_backend) == (256, 4, 10.0, 4.0, 16, "cpu")
    assert a.cache_bytes is None and a.reserve_bytes is None
