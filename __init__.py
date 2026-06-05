try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ModuleNotFoundError as exc:
    # Unit-test environments do not provide ComfyUI's `folder_paths`/`comfy`
    # modules.  Keep package importable so tests for standalone scripts can run;
    # real ComfyUI loads with those dependencies present and follows the normal
    # import path above.
    if exc.name not in {"folder_paths", "comfy"}:
        raise
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
