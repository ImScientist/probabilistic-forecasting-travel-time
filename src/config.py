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


@dataclass
class ModelConfig:
    """ Architecture configuration shared by all model wrappers (the content
    of ``default_args.model_args``). The input feature columns are held in a
    nested :class:`FeatureGroups`. """

    features: FeatureGroups = field(default_factory=FeatureGroups)
    embedding_dim: int = 10
    layer_sizes: tuple | list = (32, 32, 8)
    l2: float = .001
    dropout: float = 0
    dropout_min_layer_size: int = 12
    batch_normalization: bool = False

    @classmethod
    def from_dict(cls, data: "ModelConfig | dict | None") -> "ModelConfig":
        """ Build from a plain dict (e.g. parsed from JSON args), passing
        through an existing instance unchanged and mapping ``None`` to the
        defaults. Unknown keys are ignored; a nested ``features`` dict is
        turned into a :class:`FeatureGroups`. """

        if isinstance(data, cls):
            return data

        if data is None:
            return cls()

        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}

        if 'features' in kwargs:
            kwargs['features'] = FeatureGroups.from_dict(kwargs['features'])

        return cls(**kwargs)

    def to_dict(self) -> dict:
        """ Plain, json-serialisable dict (the nested FeatureGroups is
        recursively converted). """

        return asdict(self)
