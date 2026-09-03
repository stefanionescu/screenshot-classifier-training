"""Repository configuration-boundary policy."""

CONFIG_IMPORT_ROOTS = {
    "__future__",
    "collections.abc",
    "quality.config",
    "src.config",
    "src.state",
    "typing",
}

DECLARATIVE_EXCLUSIONS: tuple[str, ...] = ()
