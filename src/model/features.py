from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields


@dataclass
class FeatureGroups:
    """ Names of the input feature columns, grouped by how the model encodes
    them:

    - num_feats:     numeric  -> normalized
    - cat_int_feats: integer  -> one-hot encoded
    - cat_str_feats: string   -> one-hot encoded
    - emb_int_feats: integer  -> embedding
    - emb_str_feats: string   -> embedding
    """

    num_feats: list[str] = field(default_factory=list)
    cat_int_feats: list[str] = field(default_factory=list)
    cat_str_feats: list[str] = field(default_factory=list)
    emb_int_feats: list[str] = field(default_factory=list)
    emb_str_feats: list[str] = field(default_factory=list)

    @classmethod
    def field_names(cls) -> set[str]:
        """ Names of the feature-group fields. """

        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, data: "FeatureGroups | dict | None") -> "FeatureGroups":
        """ Build from a plain dict (e.g. parsed from JSON args), passing
        through an existing instance unchanged and mapping ``None`` to an
        empty set of groups. Only the recognised feature-group keys are used. """

        if isinstance(data, cls):
            return data

        if data is None:
            return cls()

        known = cls.field_names()
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)
