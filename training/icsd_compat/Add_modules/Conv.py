"""Re-export shim: get_conv2d/autopad are defined in models.nn.modules.conv,
which is fully imported before any Add_modules import site executes."""

from models.nn.modules.conv import autopad, get_conv2d  # noqa: F401
