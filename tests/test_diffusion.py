"""Executable specification for :mod:`sss.diffusion`.

15 tests, all red while the eight public functions are stubs:

* ten tests, one per known defect (1-10);
* five acceptance tests: profile peak at ``r == 0``, numerical profile
  integral of 1, empty material, ``n == 0`` in a batch, batch key order.

Initial state: ``python -m pytest tests -q`` -> ``15 failed``.
"""

from __future__ import annotations

import math

import pytest

from sss.diffusion import (
    SssError,
    apply_sss,
    blur_weight,
    diffusion_profile,
    fresnel,
    sample_offsets,
    sss_batch,
    total_profile,
    validate,
)

SKIN = {"albedo": (0.90, 0.55, 0.40), "radius": 2.0, "ior": 1.45}


def _integrate_profile(radius: float, steps: int = 30000, factor: float = 30.0) -> float:
    """Midpoint-rule integral of the profile over ``[0, factor * radius]``."""
    h = factor * radius / steps
    total = 0.0
    for i in range(steps):
        total += diffusion_profile((i + 0.5) * h, radius)
    return total * h


# --------------------------------------------------------------------------
# One test per known defect.
# --------------------------------------------------------------------------


def test_profile_peak_normalisation() -> None:
    """Defect 1: the profile must be normalised, energy must not drift."""
    for radius in (0.25, 1.0, 4.0):
        assert diffusion_profile(0.0, radius) == pytest.approx(
            1.0 / radius, abs=1e-6
        )
        assert diffusion_profile(radius, radius) == pytest.approx(
            math.exp(-1.0) / radius, abs=1e-6
        )


def test_total_profile_uses_per_channel_radii() -> None:
    """Defect 2: R, G and B must keep their own scattering radii."""
    radii = (0.4, 1.1, 2.7)
    r = 0.8
    weights = (0.2126, 0.7152, 0.0722)
    expected = sum(
        weight * diffusion_profile(r, radius)
        for weight, radius in zip(weights, radii)
    )
    assert total_profile(radii, r) == pytest.approx(expected, abs=1e-6)

    # Weights sum to 1, so equal radii collapse to the single-channel profile.
    assert total_profile((1.3, 1.3, 1.3), r) == pytest.approx(
        diffusion_profile(r, 1.3), abs=1e-6
    )

    # Changing only the blue radius must change the result.
    collapsed_blue = total_profile((0.4, 1.1, 1.1), r)
    assert abs(total_profile(radii, r) - collapsed_blue) > 1e-6


def test_sample_offsets_are_origin_centred() -> None:
    """Defect 3: offsets must be symmetric, not biased to one side."""
    n, radius, seed = 256, 2.0, 12345
    offsets = sample_offsets(n, radius, seed)

    assert len(offsets) == n
    xs = [offset[0] for offset in offsets]
    ys = [offset[1] for offset in offsets]
    assert all(-radius - 1e-9 <= x <= radius + 1e-9 for x in xs)
    assert all(-radius - 1e-9 <= y <= radius + 1e-9 for y in ys)
    assert min(xs) < 0.0 < max(xs)
    assert min(ys) < 0.0 < max(ys)
    assert sum(xs) / n == pytest.approx(0.0, abs=0.2 * radius)
    assert sum(ys) / n == pytest.approx(0.0, abs=0.2 * radius)

    # Same arguments -> same list; a different seed -> a different list.
    assert sample_offsets(n, radius, seed) == offsets
    assert sample_offsets(n, radius, seed + 1) != offsets


def test_blur_weight_decays_with_distance() -> None:
    """Defect 4: weights must fall off, otherwise edges brighten."""
    radius = 2.0
    assert blur_weight(0.0, radius) == pytest.approx(1.0, abs=1e-6)

    weights = [
        blur_weight(factor * radius, radius) for factor in (0.25, 1.0, 2.0, 4.0)
    ]
    assert all(0.0 <= weight <= 1.0 for weight in weights)
    assert weights[0] > weights[1] > weights[2] > weights[3]


def test_thicker_layers_transmit_less() -> None:
    """Defect 5: a thicker slab must let less light through."""
    color = (1.0, 0.8, 0.6)
    thin = apply_sss(color, 0.4, SKIN)
    thick = apply_sss(color, 6.0, SKIN)

    assert len(thin) == 3
    assert len(thick) == 3
    for thin_c, thick_c in zip(thin, thick):
        assert 0.0 <= thin_c <= 1.0
        assert 0.0 <= thick_c <= 1.0
        assert thin_c > thick_c

    expected = tuple(
        c * albedo * math.exp(-0.4 / SKIN["radius"])
        for c, albedo in zip(color, SKIN["albedo"])
    )
    assert thin == pytest.approx(expected, abs=1e-6)


def test_fresnel_changes_with_view_angle() -> None:
    """Defect 6: a constant Fresnel term kills grazing-angle highlights."""
    f0 = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2  # 0.04
    assert fresnel(1.0, 1.5) == pytest.approx(f0, abs=1e-6)
    assert fresnel(0.0, 1.5) == pytest.approx(1.0, abs=1e-6)

    values = [fresnel(cos_theta, 1.5) for cos_theta in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a > b for a, b in zip(values, values[1:]))

    assert fresnel(0.0, 2.0) == pytest.approx(1.0, abs=1e-6)
    assert fresnel(1.0, 2.0) == pytest.approx((1.0 / 3.0) ** 2, abs=1e-6)


def test_invalid_ior_is_rejected() -> None:
    """Defect 7: ior <= 1 must raise SssError before any division."""
    for bad_ior in (1.0, 0.5, -1.0, -2.0):
        with pytest.raises(SssError):
            fresnel(0.5, bad_ior)

    problems = validate({"albedo": (0.8, 0.5, 0.4), "radius": 1.0, "ior": 1.0})
    assert problems
    assert any("ior" in problem.lower() for problem in problems)


def test_out_of_range_albedo_is_clamped() -> None:
    """Defect 8: albedo outside [0, 1] must not leak out of the result."""
    material = {"albedo": (1.5, -0.2, 0.5), "radius": 1.0, "ior": 1.5}

    problems = validate(material)
    assert problems
    assert any("albedo" in problem.lower() for problem in problems)

    result = apply_sss((1.0, 1.0, 1.0), 0.0, material)
    assert result == pytest.approx((1.0, 0.0, 0.5), abs=1e-6)
    assert all(0.0 <= component <= 1.0 for component in result)


def test_batch_keeps_per_material_sampling() -> None:
    """Defect 9: materials must not share one sample set."""
    materials = {
        "skin": {"albedo": (0.90, 0.55, 0.40), "radius": 1.2, "ior": 1.45},
        "wax": {"albedo": (0.85, 0.80, 0.70), "radius": 3.5, "ior": 1.44},
    }
    config = {"n": 8, "seed": 99, "color": (1.0, 0.8, 0.6), "thickness": 2.0}

    result = sss_batch(materials, config)

    assert set(result) == set(materials)
    assert result["skin"]["offsets"] == sample_offsets(8, 1.2, 99)
    assert result["wax"]["offsets"] == sample_offsets(8, 3.5, 100)
    assert result["skin"]["offsets"] != result["wax"]["offsets"]

    for name, material in materials.items():
        assert result[name]["color"] == pytest.approx(
            apply_sss(config["color"], config["thickness"], material), abs=1e-6
        )


def test_sample_offsets_rejects_zero_count() -> None:
    """Defect 10: n = 0 must raise SssError instead of dividing by zero."""
    with pytest.raises(SssError):
        sample_offsets(0, 1.0, 7)
    with pytest.raises(SssError):
        sample_offsets(-4, 1.0, 7)


# --------------------------------------------------------------------------
# Five acceptance / combination tests.
# --------------------------------------------------------------------------


def test_profile_maximum_at_zero() -> None:
    """The profile peaks at r = 0."""
    radius = 2.0
    peak = diffusion_profile(0.0, radius)
    for r in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        assert diffusion_profile(r, radius) <= peak
    assert peak > diffusion_profile(0.01, radius)


def test_profile_numerical_integral_is_one() -> None:
    """The normalised profile integrates to 1 within ±0.01."""
    assert _integrate_profile(1.0) == pytest.approx(1.0, abs=0.01)


def test_validate_rejects_empty_material() -> None:
    """An empty (or incomplete) material must raise SssError."""
    with pytest.raises(SssError):
        validate({})
    with pytest.raises(SssError):
        validate({"albedo": (0.9, 0.5, 0.4)})


def test_batch_rejects_zero_samples() -> None:
    """A batch with n = 0 fails atomically instead of dividing by zero."""
    materials = {"only": dict(SKIN)}
    config = {"n": 0, "seed": 3, "color": (1.0, 1.0, 1.0), "thickness": 1.0}
    with pytest.raises(SssError):
        sss_batch(materials, config)


def test_batch_preserves_material_order() -> None:
    """A successful batch keeps the input keys and insertion order."""
    materials = {
        "c": {"albedo": (0.9, 0.5, 0.4), "radius": 1.0, "ior": 1.45},
        "a": {"albedo": (0.8, 0.7, 0.6), "radius": 2.0, "ior": 1.40},
        "b": {"albedo": (0.7, 0.6, 0.5), "radius": 3.0, "ior": 1.50},
    }
    config = {"n": 4, "seed": 11, "color": (1.0, 0.9, 0.8), "thickness": 1.5}

    result = sss_batch(materials, config)

    assert list(result.keys()) == ["c", "a", "b"]
    for name in ("c", "a", "b"):
        assert set(result[name].keys()) == {"offsets", "color"}
        assert len(result[name]["offsets"]) == 4
        assert len(result[name]["color"]) == 3
        assert all(0.0 <= component <= 1.0 for component in result[name]["color"])
