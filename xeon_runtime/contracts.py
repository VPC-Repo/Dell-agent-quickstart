"""Data-only contracts. No remote schema references, Python expressions, or eval."""

import json

from jsonschema import Draft202012Validator


def check_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a JSON object")
    if len(json.dumps(schema)) > 20000:
        raise ValueError("Schema exceeds 20 KB")

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("$ref", "$dynamicRef"):
                    raise ValueError(
                        "Schema references are not supported; use an inline schema"
                    )
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception:
        raise ValueError("Invalid JSON Schema") from None


def validate_data(value, schema, label="Output"):
    check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value), key=lambda e: str(list(e.path))
    )
    if errors:
        path = ".".join(str(x) for x in errors[0].path) or "$"
        raise ValueError(f"{label} contract failed at {path}: {errors[0].validator}")
    return value


def as_data(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def field(value, path):
    value = as_data(value)
    if not path or path == "$":
        return value
    for part in path.removeprefix("$.").split("."):
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError(f"Input mapping cannot resolve field {path}") from None
    return value


def mapped_input(mapping, outputs, query):
    result = {}
    for name, binding in mapping.items():
        source = binding["source"]
        value = query if source == "$input" else outputs.get(source)
        if source != "$input" and source not in outputs:
            raise ValueError(f"Input mapping source {source} did not execute")
        result[name] = field(value, binding.get("path", ""))
    return result


def predicate(value, operator, expected):
    if operator == "exists":
        return value is not None
    if operator == "equals":
        return type(value) is type(expected) and value == expected
    if operator == "not_equals":
        return not predicate(value, "equals", expected)
    if operator == "contains":
        if isinstance(value, (str, dict)) and not isinstance(expected, str):
            raise ValueError(
                "Condition contains requires a text value for text or object inputs"
            )
        return isinstance(value, (str, list, dict)) and expected in value
    if operator in ("gt", "gte", "lt", "lte"):
        if type(value) not in (int, float) or type(expected) not in (int, float):
            raise ValueError("Condition comparison requires numeric values")
        return {
            "gt": value > expected,
            "gte": value >= expected,
            "lt": value < expected,
            "lte": value <= expected,
        }[operator]
    raise ValueError("Unsupported condition operator")
