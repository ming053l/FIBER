from .extractor import Extractor, Teacher
from .spatial import SpatialExtractor, SpatialTeacher

TEACHERS = {"resnet18": Teacher, "spatial": SpatialTeacher}
EXTRACTORS = {"resnet18": Extractor, "spatial": SpatialExtractor}


def _accepted(cls, kw: dict) -> dict:
    """Keep only the kwargs the class actually takes, so callers can pass the union
    (d, latent_shape, k, ...) without knowing which architecture they will get."""
    import inspect
    names = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {k: v for k, v in kw.items() if k in names}


def build_teacher(arch: str = "resnet18", **kw):
    """The teacher class is part of what the certified operator certifies, so the
    architecture is named in every report (P0-5)."""
    if arch not in TEACHERS:
        raise KeyError(f"unknown teacher arch {arch!r}; have {sorted(TEACHERS)}")
    cls = TEACHERS[arch]
    return cls(**_accepted(cls, kw))


def build_extractor(arch: str = "resnet18", **kw):
    if arch not in EXTRACTORS:
        raise KeyError(f"unknown extractor arch {arch!r}; have {sorted(EXTRACTORS)}")
    cls = EXTRACTORS[arch]
    return cls(**_accepted(cls, kw))


__all__ = ["Extractor", "Teacher", "SpatialTeacher", "SpatialExtractor",
           "build_teacher", "build_extractor", "TEACHERS", "EXTRACTORS"]
