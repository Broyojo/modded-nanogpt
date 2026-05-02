from importlib import import_module


def load_config(name: str = "baseline"):
    """Import jepa.configs.<name> and return its CONFIG object."""
    return import_module(f"jepa.configs.{name}").CONFIG
