"""Errors evalkit raises. Every one carries a message a user can act on."""
from __future__ import annotations


class EvalkitError(Exception):
    """Base class for all evalkit errors."""


class ConfigError(EvalkitError):
    """The configuration is invalid (missing key, wrong type, unknown option)."""


class PluginNotFoundError(ConfigError):
    """No plugin is registered under that category and name."""


class PluginConfigError(ConfigError):
    """A plugin's configuration block is invalid."""


class PluginDependencyError(EvalkitError):
    """A plugin needs an optional package that is not installed."""

    def __init__(self, plugin: str, missing: str, extra: str | None):
        hint = f"pip install 'evalkit[{extra}]'" if extra else f"pip install {missing}"
        super().__init__(f"plugin '{plugin}' needs the package '{missing}', which is not installed. "
                         f"Install it with: {hint}")
        self.plugin, self.missing, self.extra = plugin, missing, extra


class SecretNotFoundError(EvalkitError):
    """A ${secret:...} reference could not be resolved."""


class BudgetExceededError(EvalkitError):
    """The run hit its configured token or cost cap."""


class AccessDeniedError(EvalkitError):
    """The caller is not allowed to perform this action (RBAC)."""
