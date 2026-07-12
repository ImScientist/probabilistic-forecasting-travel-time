from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields

from model.features import FeatureGroups


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
