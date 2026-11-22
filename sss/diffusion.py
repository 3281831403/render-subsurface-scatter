"""Subsurface scattering (SSS) core for a skin shader.

Conventions
-----------
* All distances and radii are millimetres (mm).
* All colours (``color``, ``albedo`` and returned RGB tuples) are floats in
  the closed range ``[0.0, 1.0]``.
* Expected numeric agreement with this contract is ``1e-6`` unless a
  docstring states a different tolerance.

``tests/test_diffusion.py`` is the executable version of the contract.
"""

from __future__ import annotations

import math
import numbers
import random
from collections.abc import Mapping, Sequence

__all__ = [
    "SssError",
    "diffusion_profile",
    "total_profile",
    "sample_offsets",
    "blur_weight",
    "apply_sss",
    "fresnel",
    "validate",
    "sss_batch",
]

RGB_WEIGHTS = (0.2126, 0.7152, 0.0722)
"""Rec. 709 luminance weights used to combine the RGB diffusion profiles."""

_MATERIAL_KEYS = ("albedo", "radius", "ior")
_BATCH_KEYS = ("n", "seed", "color", "thickness")


class SssError(Exception):
    """Raised when a material, a parameter or a batch configuration is invalid."""


# ---------------------------------------------------------------------------
# Internal validation helpers.
# ---------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    """True for real numbers, excluding booleans."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _as_number(value: object, name: str) -> float:
    """Return *value* as a float or raise :class:`SssError`."""
    if not _is_number(value):
        raise SssError(f"{name} must be a number, got {value!r}")
    return float(value)


def _non_negative(value: object, name: str) -> float:
    """Return *value* as a float ``>= 0`` or raise :class:`SssError`."""
    number = _as_number(value, name)
    if not number >= 0.0:  # also rejects NaN
        raise SssError(f"{name} must be >= 0, got {number!r}")
    return number


def _positive(value: object, name: str) -> float:
    """Return *value* as a float ``> 0`` or raise :class:`SssError`."""
    number = _as_number(value, name)
    if not number > 0.0:  # also rejects NaN
        raise SssError(f"{name} must be > 0, got {number!r}")
    return number


def _as_components(value: object, name: str) -> tuple[float, float, float]:
    """Return a three-number sequence as a float tuple or raise :class:`SssError`."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SssError(f"{name} must be a sequence of three numbers, got {value!r}")
    components = tuple(value)
    if len(components) != 3:
        raise SssError(f"{name} must contain exactly three numbers, got {len(components)}")
    for index, component in enumerate(components):
        if not _is_number(component):
            raise SssError(f"{name}[{index}] must be a number, got {component!r}")
    return components  # type: ignore[return-value]


def _clamp01(value: float) -> float:
    """Clamp *value* to the closed unit interval."""
    return min(1.0, max(0.0, value))


def _require_mapping(value: object, name: str, keys: tuple[str, ...]) -> Mapping:
    """Return *value* if it is a mapping holding *keys*; otherwise raise."""
    if not isinstance(value, Mapping):
        raise SssError(f"{name} must be a mapping, got {type(value).__name__}")
    for key in keys:
        if key not in value:
            raise SssError(f"{name} is missing the {key!r} entry")
    return value


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def diffusion_profile(r: float, radius: float) -> float:
    """Normalised radial diffusion profile of a single colour channel.

    Parameters
    ----------
    r:
        Distance from the shading point in millimetres; ``r >= 0``.
    radius:
        Scattering radius of the channel in millimetres; ``radius > 0``.

    Returns
    -------
    float
        ``math.exp(-r / radius) / radius``.

    The profile is normalised in one radial dimension::

        integral_0_inf diffusion_profile(x, radius) dx == 1.0

    Its maximum is at ``r == 0`` with
    ``diffusion_profile(0, radius) == 1.0 / radius``, and it decreases
    monotonically as ``r`` grows.

    Raises
    ------
    SssError
        If ``r < 0`` or ``radius <= 0``.
    """
    distance = _non_negative(r, "r")
    scatter_radius = _positive(radius, "radius")
    return math.exp(-distance / scatter_radius) / scatter_radius


def total_profile(
    radii: tuple[float, float, float] | list[float], r: float
) -> float:
    """Weighted RGB combination of three per-channel diffusion profiles.

    ``radii`` is a sequence of exactly three positive scattering radii in
    millimetres, ordered red, green, blue.  The channels are blended with
    the Rec. 709 luminance weights ``RGB_WEIGHTS = (0.2126, 0.7152,
    0.0722)``::

        sum(RGB_WEIGHTS[c] * diffusion_profile(r, radii[c]) for c in range(3))

    Because the weights sum to 1, three equal radii reproduce
    ``diffusion_profile(r, radius)`` within ``1e-6``.

    Raises ``SssError`` if ``r < 0``, ``len(radii) != 3`` or any radius is
    not strictly positive.
    """
    distance = _non_negative(r, "r")
    channels = _as_components(radii, "radii")
    for index, channel_radius in enumerate(channels):
        if channel_radius <= 0.0:
            raise SssError(f"radii[{index}] must be > 0, got {channel_radius!r}")
    return sum(
        weight * diffusion_profile(distance, channel_radius)
        for weight, channel_radius in zip(RGB_WEIGHTS, channels)
    )


def sample_offsets(n: int, radius: float, seed: int) -> list[tuple[float, float]]:
    """Deterministic, origin-centred 2-D sample offsets.

    Returns ``n`` ``(x, y)`` offsets, each coordinate confined to
    ``[-radius, radius]`` (a square kernel).  Sampling is symmetric around
    the origin: for a large ``n`` the mean of each coordinate approaches
    ``0`` and both negative and positive values occur.  A purely positive
    offset set is incorrect.

    The result depends only on ``(n, radius, seed)``: equal arguments give
    equal lists and different seeds give different lists.  Global random
    state must not be used.

    Raises ``SssError`` if ``n <= 0`` or ``radius <= 0``.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise SssError(f"n must be an integer, got {n!r}")
    if n <= 0:
        raise SssError(f"n must be > 0, got {n!r}")
    scatter_radius = _positive(radius, "radius")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SssError(f"seed must be an integer, got {seed!r}")

    rng = random.Random(seed)
    return [
        (
            rng.uniform(-scatter_radius, scatter_radius),
            rng.uniform(-scatter_radius, scatter_radius),
        )
        for _ in range(n)
    ]


def blur_weight(dist: float, radius: float) -> float:
    """Radial falloff weight of one sample.

    ``dist`` is the sample distance in millimetres (``>= 0``) and
    ``radius`` the channel radius in millimetres (``> 0``).

    The weight is in ``[0, 1]``, is ``1`` at ``dist == 0`` and strictly
    decreases with distance, e.g.::

        blur_weight(0.5 * radius, radius) > blur_weight(radius, radius)
        blur_weight(radius, radius) > blur_weight(2 * radius, radius)

    ``math.exp(-dist / radius)`` is a valid implementation.

    Raises ``SssError`` if ``dist < 0`` or ``radius <= 0``.
    """
    distance = _non_negative(dist, "dist")
    scatter_radius = _positive(radius, "radius")
    return math.exp(-distance / scatter_radius)


def apply_sss(
    color: tuple[float, float, float],
    thickness: float,
    material: dict,
) -> tuple[float, float, float]:
    """Colour transmitted through a slab of skin.

    ``color`` is the incoming RGB colour (three floats in ``[0, 1]``),
    ``thickness`` the slab thickness in millimetres (``>= 0``) and
    ``material`` a mapping ``{"albedo": (r, g, b), "radius": float,
    "ior": float}``.

    The per-channel result is::

        clamp(color[c] * albedo[c] * math.exp(-thickness / radius), 0, 1)

    so a thicker slab transmits less and every returned component lies in
    ``[0, 1]``.  Albedo components outside ``[0, 1]`` are clamped by this
    formula (use :func:`validate` for strict detection); they do not make
    this function raise.

    Raises ``SssError`` if ``thickness < 0``, the material is missing one
    of its keys, ``albedo`` is not three numeric components or
    ``radius <= 0``.
    """
    slab_thickness = _non_negative(thickness, "thickness")
    material = _require_mapping(material, "material", _MATERIAL_KEYS)
    albedo = _as_components(material["albedo"], "albedo")
    scatter_radius = _positive(material["radius"], "radius")
    incoming = _as_components(color, "color")

    attenuation = math.exp(-slab_thickness / scatter_radius)
    return tuple(
        _clamp01(component * albedo_component * attenuation)
        for component, albedo_component in zip(incoming, albedo)
    )


def fresnel(cos_theta: float, ior: float) -> float:
    """Schlick approximation of dielectric Fresnel reflectance.

    ``cos_theta`` is the cosine of the view angle against the surface
    normal, in ``[0, 1]``; ``ior`` is the index of refraction of the
    material (``> 1``, air is 1.0).  With
    ``f0 = ((ior - 1) / (ior + 1)) ** 2``::

        f0 + (1 - f0) * (1 - cos_theta) ** 5

    Normal incidence returns ``f0`` and grazing incidence
    (``cos_theta == 0``) returns ``1``.  A constant return value is
    incorrect.

    Raises ``SssError`` if ``ior <= 1`` (checked before any division) or
    ``cos_theta`` is outside ``[0, 1]``.
    """
    # Validate the denominator first: a bad ior must never reach a division.
    if not _is_number(ior) or not ior > 1.0:  # `not >` also rejects NaN
        raise SssError(f"ior must be a number > 1, got {ior!r}")
    if not _is_number(cos_theta) or not 0.0 <= cos_theta <= 1.0:
        raise SssError(f"cos_theta must be in [0, 1], got {cos_theta!r}")

    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5


def validate(material: dict) -> list[str]:
    """List the problems of a material.

    ``material`` is ``{"albedo": (r, g, b), "radius": float, "ior":
    float}``.  The returned list is empty when the material is valid and
    otherwise holds one human-readable string per problem, in this order:

    * radius: must be a number ``> 0`` mm;
    * albedo: each component must be a number in ``[0, 1]``;
    * ior: must be a number ``> 1``.

    Each message names the offending field, i.e. contains ``radius``,
    ``albedo`` or ``ior``.

    Structural problems raise ``SssError`` instead of being returned:
    ``material`` not being a mapping, a missing key, an albedo that is not
    three numeric components, or a non-numeric radius/ior.  In particular
    ``validate({})`` raises ``SssError``.

    Always returns ``list[str]`` when it returns (possibly empty).
    """
    material = _require_mapping(material, "material", _MATERIAL_KEYS)
    albedo = _as_components(material["albedo"], "albedo")
    radius = material["radius"]
    ior = material["ior"]
    if not _is_number(radius):
        raise SssError(f"radius must be a number, got {radius!r}")
    if not _is_number(ior):
        raise SssError(f"ior must be a number, got {ior!r}")

    problems: list[str] = []
    if not radius > 0:
        problems.append(f"radius must be greater than 0 mm, got {radius!r}")
    for index, component in enumerate(albedo):
        if not 0.0 <= component <= 1.0:
            problems.append(f"albedo[{index}] must be in [0, 1], got {component!r}")
    if not ior > 1:
        problems.append(f"ior must be greater than 1, got {ior!r}")
    return problems


def sss_batch(materials: dict, config: dict) -> dict:
    """Evaluate several materials atomically and without cross-talk.

    ``materials`` is an insertion-ordered mapping ``name -> material``.
    ``config`` is ``{"n": int, "seed": int, "color": (r, g, b),
    "thickness": float}`` with ``n > 0``, ``color`` in ``[0, 1]`` and
    ``thickness >= 0``.

    For the material at insertion index ``i`` the batch uses its own
    sample set::

        sample_offsets(config["n"], material["radius"], config["seed"] + i)

    and its own transmitted colour::

        apply_sss(config["color"], config["thickness"], material)

    Sharing one sample set between different materials is incorrect.

    Returns ``{name: {"offsets": [...], "color": (r, g, b)}}`` with the
    same keys as ``materials`` and the same insertion order.

    The batch is atomic: every material is validated first (see
    :func:`validate`); if any material is invalid, ``SssError`` is raised
    and no partial result is produced.  Also raises ``SssError`` when
    ``materials`` is empty, ``config`` is malformed or ``n <= 0``.
    """
    if not isinstance(materials, Mapping):
        raise SssError(f"materials must be a mapping, got {type(materials).__name__}")
    if not materials:
        raise SssError("materials must not be empty")

    config = _require_mapping(config, "config", _BATCH_KEYS)

    n = config["n"]
    if not isinstance(n, int) or isinstance(n, bool):
        raise SssError(f"config['n'] must be an integer, got {n!r}")
    if n <= 0:
        raise SssError(f"config['n'] must be > 0, got {n!r}")

    seed = config["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SssError(f"config['seed'] must be an integer, got {seed!r}")

    color = _as_components(config["color"], "config['color']")
    if any(not 0.0 <= component <= 1.0 for component in color):
        raise SssError(f"config['color'] components must be in [0, 1], got {color!r}")

    thickness = _non_negative(config["thickness"], "config['thickness']")

    # Validate every material before computing anything: the batch is atomic.
    failures = []
    for name, material in materials.items():
        problems = validate(material)
        if problems:
            failures.append(f"{name!r}: {'; '.join(problems)}")
    if failures:
        raise SssError("invalid material(s): " + " | ".join(failures))

    result: dict = {}
    for index, (name, material) in enumerate(materials.items()):
        # Each material gets its own sample set, offset by its insertion index.
        result[name] = {
            "offsets": sample_offsets(n, material["radius"], seed + index),
            "color": apply_sss(color, thickness, material),
        }
    return result
