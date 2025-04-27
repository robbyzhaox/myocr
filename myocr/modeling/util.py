import importlib
import logging

logger = logging.getLogger(__name__)


def check_module_name(module_name: str, module_map: dict):
    support_dict = list(module_map.keys())
    assert module_name in support_dict, RuntimeError(f"module {module_name} only supported")


def build_module(module_path: str, config: dict):
    module_name = config.pop("name")
    module = importlib.import_module(module_path, package="myocr.modeling")
    module_class = getattr(module, module_name)
    return module_class(**config)


def build_component(component_type, config):
    components_map = {}
    if component_type == "Backbone":
        from .backbones import BACKBONES_MAP

        components_map = BACKBONES_MAP
    elif component_type == "Transform":
        from .transforms import TRANSFORMS_MAP

        components_map = TRANSFORMS_MAP
    elif component_type == "Neck":
        from .necks import NECKS_MAP

        components_map = NECKS_MAP
    elif component_type == "Head":
        from .heads import HEADS_MAP

        components_map = HEADS_MAP

    component_name = config.get("name")
    check_module_name(component_name, components_map)
    head_path = components_map.get(component_name)
    return build_module(head_path, config)  # type: ignore
