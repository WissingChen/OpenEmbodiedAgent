# Track 2: Desktop Pet Robot — Physical Interaction

> Owner: TBD | Status: Planning | Priority: Medium

## 1. Objective

Build a driver for a low-cost desktop robot (OEA) that provides physical emotional expression and simple interaction capabilities. This is the Tier 3 user-facing embodiment — the "alive pet" that chats, nods, shakes its head, and points at objects.

## 2. Scope

### In Scope
- Implement `DesktopPetDriver(BaseDriver)` in `hal/drivers/desktop_pet_driver.py`
- Serial/USB communication with the physical desktop robot's microcontroller
- Expression actions: `nod_head`, `shake_head`, `look_at`, `express_emotion`
- Simple pointing: `point_to` with pan/tilt servo control
- LED/buzzer feedback for emotional expression
- Write `hal/profiles/desktop_pet.md` describing the pet's capabilities
- `SOUL.md` integration — the pet's personality drives its proactive behavior

### Out of Scope
- Object manipulation (no gripper)
- Autonomous navigation (stationary robot)
- Vision processing (handled by MCP Vision Server in Track A)

## 3. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `hal/drivers/desktop_pet_driver.py` | Create | `DesktopPetDriver(BaseDriver)` implementation |
| `hal/profiles/desktop_pet.md` | Create | EMBODIED.md profile for the desktop pet |
| `tests/test_desktop_pet_driver.py` | Create | Driver tests (with hardware mock) |

## 4. EMBODIED.md Profile Requirements

- Robot name and type (desktop pet)
- Degrees of freedom (typically 2-DOF pan/tilt head)
- Servo ranges and speed limits
- Supported actions: `nod_head`, `shake_head`, `point_to`, `express_emotion`
- Communication interface (serial port, baud rate)
- No manipulation capability — explicitly state this so the Critic Agent can reject pick/place actions

## 5. Action Specification

| Action | Parameters | Expected Behavior |
|--------|-----------|-------------------|
| `nod_head` | `intensity: float` (0-1) | Tilt head down then up |
| `shake_head` | `intensity: float` (0-1) | Pan head left-right-left |
| `point_to` | `direction: str` or `x, y` | Pan/tilt toward direction |
| `look_at` | `target: str` | Orient head toward named object (requires ENVIRONMENT.md lookup) |
| `express_emotion` | `emotion: str` (happy, sad, curious, surprised) | LED pattern + head gesture combo |

## 6. Milestones & Acceptance Criteria

### Milestone M1: Driver Scaffold
- [ ] `DesktopPetDriver` class exists in `hal/drivers/desktop_pet_driver.py`
- [ ] Driver registered as `"desktop_pet"` in registry
- [ ] `pytest tests/test_hal_base_driver.py -k desktop_pet` — all 10 contract tests green (mock mode)
- [ ] `hal/profiles/desktop_pet.md` exists with correct capability declaration

### Milestone M2: Mock Execution
- [ ] `nod_head({})` returns success string without raising
- [ ] `shake_head({})` returns success string
- [ ] `express_emotion({"emotion": "happy"})` returns success string
- [ ] `execute_action("pick_up", {...})` returns error string (not supported by this body)
- [ ] `get_scene()` returns a dict containing robot pose/state info

### Milestone M3: Hardware Integration (physical robot required)
- [ ] Serial/USB connection established; no connection errors on startup
- [ ] `nod_head` produces visible head motion (confirmed by eye)
- [ ] `shake_head` produces visible lateral motion
- [ ] `express_emotion` triggers distinct LED + servo patterns for ≥ 4 emotions (happy, sad, curious, surprised)
- [ ] `point_to(direction="left")` rotates head visibly to the left

### Milestone M4: Heartbeat & Proactive Behavior
- [ ] `HEARTBEAT.md` periodic task triggers robot to nod every 30 minutes when idle
- [ ] Critic correctly rejects `pick_up` / `push` actions (confirms "not supported" in rejection message)

### Milestone M5: Full Pipeline
- [ ] User types "show me happy" in CLI → Agent writes `ACTION.md` → Watchdog executes `express_emotion` on physical robot
- [ ] Robot reacts within 2 seconds of the command

## 7. Dependencies

- `pyserial` (pip install pyserial)
- Desktop pet hardware with Arduino/ESP32 firmware
- Firmware protocol documentation (to be defined by hardware owner)

## 8. Hardware-Software Protocol

The driver communicates with the microcontroller via a simple serial JSON protocol:

```json
{"cmd": "nod", "intensity": 0.8}
{"cmd": "pan", "angle": 45}
{"cmd": "led", "pattern": "happy"}
```

The MCU responds with:
```json
{"status": "ok", "position": {"pan": 45, "tilt": 0}}
```
