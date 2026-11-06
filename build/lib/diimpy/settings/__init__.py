try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

DEFAULT_SETTINGS_DIR = files(__name__)
