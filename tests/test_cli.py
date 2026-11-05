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


def test_train_denoise_lbfgs_options():
    a = train_denoise.parser().parse_args(["--data", "D", "--out", "O"])
    assert (a.optimizer, a.lbfgs_rounds, a.lbfgs_iter, a.lbfgs_history) == ("adam", 10, 50, 10)
    a = train_denoise.parser().parse_args(["--data", "D", "--out", "O", "--optimizer", "lbfgs", "--lbfgs-rounds", "3"])
    assert a.optimizer == "lbfgs" and a.lbfgs_rounds == 3


def test_train_denoise_spectrum_and_conditioning_options():
    a = train_denoise.parser().parse_args(["--data", "D", "--out", "O"])
    assert (a.bias, a.cond, a.cl_masks, a.cl_width) == ("live", "none", "0,30,45", 30)
    a = train_denoise.parser().parse_args(["--data", "D", "--out", "O", "--loss", "data=1,cl=1", "--cond", "log857",
                                           "--bias", "frozen"])
    assert a.cond == "log857" and a.bias == "frozen"


def test_train_denoise_stat_masks_option():
    assert train_denoise.parser().parse_args(["--data", "D", "--out", "O"]).stat_masks is None
    a = train_denoise.parser().parse_args(["--data", "D", "--out", "O", "--stat-masks", "0,20,30,45"])
    assert a.stat_masks == "0,20,30,45"


def test_train_denoise_eb_sigma_warmup_options():
    a = train_denoise.parser().parse_args(["--data", "D", "--out", "O"])
    assert (a.pol_stats, a.bias_warmup, a.init) == ("qu", 0, None)
    a = train_denoise.parser().parse_args(["--data", "D", "--out", "O", "--joint", "--pol-stats", "eb",
                                           "--cond", "log857+sigma", "--bias", "frozen", "--bias-warmup", "500",
                                           "--init", "run/model.pt"])
    assert (a.pol_stats, a.cond, a.bias_warmup, a.init) == ("eb", "log857+sigma", 500, "run/model.pt")
