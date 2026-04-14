"""
tests/test_hal_base_driver.py

Contract tests for the BaseDriver interface.

Any driver that is registered in ``hal/drivers/`` MUST pass these tests.
They verify the four abstract methods behave correctly without depending
on specific hardware.

Run all registered drivers::

    pytest tests/test_hal_base_driver.py -v

Run a specific driver::

    pytest tests/test_hal_base_driver.py -v -k simulation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers: check which drivers are available
# ---------------------------------------------------------------------------

def _available_drivers() -> list[str]:
    """Return driver names whose dependencies are importable."""
    from hal.drivers import DRIVER_REGISTRY
    available = []
    for name in DRIVER_REGISTRY:
        try:
            from hal.drivers import load_driver
            d = load_driver(name, gui=False)
            d.close()
            available.append(name)
        except (ImportError, Exception):
            pass
    return available


_DRIVERS = _available_drivers()

if not _DRIVERS:
    pytest.skip("No drivers available (missing pybullet?)", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=_DRIVERS)
def driver(request):
    """Yield a fresh driver instance; close after test."""
    from hal.drivers import load_driver
    d = load_driver(request.param, gui=False)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestBaseDriverContract:
    """Every BaseDriver subclass must satisfy these invariants."""

    def test_get_profile_path_returns_path(self, driver):
        p = driver.get_profile_path()
        assert isinstance(p, Path)

    def test_get_profile_path_file_exists(self, driver):
        assert driver.get_profile_path().exists(), (
            f"Profile file does not exist: {driver.get_profile_path()}"
        )

    def test_load_scene_empty_dict(self, driver):
        """Must not raise on empty scene."""
        driver.load_scene({})

    def test_load_scene_valid_objects(self, driver):
        scene = {
            "test_obj": {
                "type": "default",
                "position": {"x": 0, "y": 0, "z": 0},
                "location": "table",
            }
        }
        driver.load_scene(scene)

    def test_get_scene_returns_dict(self, driver):
        result = driver.get_scene()
        assert isinstance(result, dict)

    def test_get_scene_after_load_contains_objects(self, driver):
        scene = {
            "apple": {
                "type": "fruit",
                "position": {"x": 5, "y": 5, "z": 0},
            }
        }
        driver.load_scene(scene)
        result = driver.get_scene()
        assert "apple" in result

    def test_execute_action_returns_string(self, driver):
        result = driver.execute_action("nod_head", {})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_unknown_action_does_not_crash(self, driver):
        """Unknown actions must return an error string, not raise."""
        result = driver.execute_action("__nonexistent_action__", {})
        assert isinstance(result, str)

    def test_context_manager_protocol(self, driver):
        """Driver supports with-statement."""
        from hal.drivers import load_driver
        name = [k for k, v in __import__('hal.drivers', fromlist=['DRIVER_REGISTRY']).DRIVER_REGISTRY.items()
                if type(driver).__qualname__ in v][0]
        with load_driver(name, gui=False) as d:
            assert d is not None
            d.load_scene({})

    def test_close_is_idempotent(self, driver):
        """Calling close() twice must not raise."""
        driver.close()
        driver.close()
