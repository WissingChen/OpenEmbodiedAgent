# Track 7: Framework Core — BaseDriver, Watchdog & Context Engine

> Owner: TBD | Status: Planning | Priority: Critical (blocks all other tracks)

## 1. Objective

Build and maintain the **framework infrastructure** that all other tracks depend on: the `BaseDriver` abstract interface, the dynamic driver-loading `hal_watchdog.py`, and the enhanced `context.py` that auto-discovers workspace Markdown files. This track is the "glue" that keeps the system decoupled.

## 2. Scope

### In Scope
- Define and stabilize `hal/base_driver.py` — the abstract `BaseDriver` class
- Refactor `hal/hal_watchdog.py` to support `--driver <name>` dynamic loading
- Implement `--profile` auto-copy from `hal/profiles/{driver}.md` to workspace `EMBODIED.md`
- Refactor `OEA/agent/context.py` to dynamically discover workspace `.md` files
- Create the `hal/drivers/__init__.py` driver registry
- Write the `BaseDriver` contract test suite
- Maintain `hal/simulation/scene_io.py` as shared I/O utilities

### Out of Scope
- Implementing specific hardware drivers (Tracks 1-4)
- Orchestrator logic (Track 5)
- Memory system (Track 6)

## 3. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `hal/__init__.py` | Create | Package marker |
| `hal/base_driver.py` | Create | Abstract `BaseDriver` class |
| `hal/drivers/__init__.py` | Create | Driver registry with dynamic loading |
| `hal/hal_watchdog.py` | Modify | Add `--driver` flag and profile auto-copy |
| `OEA/agent/context.py` | Modify | Dynamic `.md` file discovery |
| `tests/test_hal_base_driver.py` | Create | Contract test suite for any driver |

## 4. BaseDriver Abstract Class Specification

```python
from abc import ABC, abstractmethod
from pathlib import Path


class BaseDriver(ABC):
    """Abstract base class for all robot body drivers.

    Every hardware/simulation embodiment MUST implement this interface.
    The HAL Watchdog loads a driver by name and interacts with it
    exclusively through these methods.
    """

    @abstractmethod
    def get_profile_path(self) -> Path:
        """Return the path to this driver's EMBODIED.md profile.

        The watchdog copies this file to the workspace on startup.
        """

    @abstractmethod
    def load_scene(self, scene: dict[str, dict]) -> None:
        """Initialize the physical/simulated world from a scene dict.

        Called once at startup with the contents of ENVIRONMENT.md.
        """

    @abstractmethod
    def execute_action(self, action_type: str, params: dict) -> str:
        """Execute a single atomic action.

        Returns a human-readable result string.
        The watchdog calls this when ACTION.md has a new command.
        """

    @abstractmethod
    def get_scene(self) -> dict[str, dict]:
        """Return the current world state as a scene dict.

        Called after each action to update ENVIRONMENT.md.
        """

    def close(self) -> None:
        """Release hardware resources. Optional override."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
```

## 5. Driver Registry Specification

`hal/drivers/__init__.py` provides a simple registry that maps driver names to classes:

```python
DRIVER_REGISTRY: dict[str, str] = {
    "simulation":  "hal.drivers.simulation_driver.SimulationDriver",
    "desktop_pet": "hal.drivers.desktop_pet_driver.DesktopPetDriver",
    "dobot_nova5": "hal.drivers.dobot_driver.DobotDriver",
    "go2_edu":     "hal.drivers.go2_driver.Go2Driver",
}

def load_driver(name: str, **kwargs) -> BaseDriver:
    """Dynamically import and instantiate a driver by name."""
```

New drivers are added by a single line in `DRIVER_REGISTRY`. No other code changes needed.

## 6. hal_watchdog.py CLI Specification

```
python hal/hal_watchdog.py [OPTIONS]

Options:
  --driver NAME     Driver name from registry (default: simulation)
  --workspace PATH  Workspace directory (default: ~/.OEA/workspace)
  --gui             Enable GUI mode (passed to driver if supported)
  --interval FLOAT  Polling interval in seconds (default: 1.0)
```

### Startup Sequence
1. Load driver class from registry by `--driver` name
2. Instantiate driver (passing `gui` flag)
3. Copy `driver.get_profile_path()` → `workspace/EMBODIED.md`
4. Read `workspace/ENVIRONMENT.md` → `driver.load_scene(scene)`
5. Enter poll loop: watch `ACTION.md` → `driver.execute_action()` → update `ENVIRONMENT.md`

## 7. context.py Enhancement Specification

### Current Behavior (problematic)
```python
BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md",
                   "EMBODIED.md", "ENVIRONMENT.md", "LESSONS.md"]
```
This is a hardcoded list. Adding `TASK.md`, `ORCHESTRATOR.md`, `MEMORY_SPATIAL.md`, or `TIMELINE.md` requires modifying core OEA code.

### Target Behavior
```python
# Core files always loaded (OEA essentials)
CORE_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]

# Embodied files loaded if present (no code change to add new ones)
EMBODIED_PATTERNS = ["EMBODIED.md", "ENVIRONMENT.md", "LESSONS.md",
                     "TASK.md", "ORCHESTRATOR.md",
                     "MEMORY_SPATIAL.md", "TIMELINE.md"]
```

The `_load_bootstrap_files()` method should:
1. Always load `CORE_FILES`
2. Scan workspace root for any `.md` matching `EMBODIED_PATTERNS`
3. Only inject files that actually exist (no error if absent)

This way Track 5 (Orchestrator) and Track 6 (Memory) can add new protocol files without touching `context.py`.

## 8. Contract Test Suite

`tests/test_hal_base_driver.py` — a parametrized test that any driver must pass:

```python
@pytest.fixture(params=["simulation"])  # Add more drivers as they're built
def driver(request):
    return load_driver(request.param, gui=False)

class TestBaseDriverContract:
    def test_get_profile_path_returns_existing_file(self, driver):
        assert driver.get_profile_path().exists()

    def test_load_scene_accepts_empty_dict(self, driver):
        driver.load_scene({})  # Must not raise

    def test_load_scene_accepts_valid_scene(self, driver):
        scene = {"obj": {"type": "default", "position": {"x": 0, "y": 0, "z": 0}}}
        driver.load_scene(scene)

    def test_execute_action_returns_string(self, driver):
        result = driver.execute_action("nod_head", {})
        assert isinstance(result, str)

    def test_get_scene_returns_dict(self, driver):
        result = driver.get_scene()
        assert isinstance(result, dict)

    def test_unknown_action_does_not_crash(self, driver):
        result = driver.execute_action("nonexistent_action_xyz", {})
        assert isinstance(result, str)  # Should return error message, not raise
```

## 9. Milestones & Acceptance Criteria

### Milestone M1: BaseDriver & Registry (no hardware)
- [ ] `hal/base_driver.py` exists, imports without error, has 4 abstract methods
- [ ] `hal/drivers/__init__.py` registry has `load_driver("simulation")` working
- [ ] Adding a new driver requires exactly: 1 new file + 1 line in `DRIVER_REGISTRY`
- [ ] `pytest tests/test_hal_base_driver.py` — all contract tests green for simulation driver

### Milestone M2: Watchdog Dynamic Loading
- [ ] `python hal/hal_watchdog.py --driver simulation` starts without error
- [ ] `hal/profiles/simulation.md` is auto-copied to `~/.nanobot/workspace/EMBODIED.md` on start
- [ ] `python hal/hal_watchdog.py --driver unknown_xyz` prints a clear error message and exits with code 1
- [ ] `python hal/hal_watchdog.py --help` lists all available drivers

### Milestone M3: Context Dynamic Scan
- [ ] `nanobot agent` loads `EMBODIED.md` from workspace when present
- [ ] Creating `TASK.md` in workspace → agent sees it in next turn (no restart required)
- [ ] Deleting `MEMORY_SPATIAL.md` → agent stops including it (no error)
- [ ] `context.py` does NOT require modification when a new protocol file is added

### Milestone M4: ROS2 Bridge (cross-cutting)
- [ ] `hal/ros2_bridge.py` exists with `subscribe(topic, callback)` and `publish(topic, msg)` stubs
- [ ] Track 4 (Go2) and Track 3 (Dobot/Piper) can import and use `ros2_bridge` without code duplication

### Milestone M5: Safety Callbacks
- [ ] `BaseDriver` has `on_collision(force_n, position)` callback (default no-op)
- [ ] Simulation driver overrides `on_collision` to write a test entry to `LESSONS.md`
- [ ] Contract test verifies `on_collision` is callable and does not raise

## 10. Dependencies

- Python ≥ 3.11
- No additional packages (framework core has zero extra deps)
