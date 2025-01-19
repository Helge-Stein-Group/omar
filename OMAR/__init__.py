from jaxtyping import install_import_hook
with install_import_hook("omar", "beartype.beartype"):
    from .omar import OMAR