from temporaldata import Data


def deep_register_fields(data: Data, field_map: dict):
    for key, value in field_map.items():
        if isinstance(value, dict):
            setattr(data, key, Data())
            deep_register_fields(getattr(data, key), value)
        else:
            setattr(data, key, value)
