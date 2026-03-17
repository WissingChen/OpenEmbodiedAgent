# Track 6: Spatiotemporal Memory System

> Owner: TBD | Status: Planning | Priority: Medium

## 1. Objective

Build a memory system that gives the robot **spatial awareness** (where things are and were) and **temporal awareness** (what happened and when). This goes beyond OEA's existing flat `MEMORY.md` by adding structured spatial indexing and a time-ordered event log.

## 2. Scope

### In Scope
- Design `MEMORY_SPATIAL.md` — a spatial index of known objects and landmarks
- Design `TIMELINE.md` — a chronological event log with timestamps
- Implement `SpatialMemoryTool` as a OEA tool for spatial queries
- Implement `TimelineTool` as a OEA tool for temporal queries
- Spatial queries: "where was the apple last seen?", "what objects are near the table?"
- Temporal queries: "what happened in the last hour?", "when did I last pick up the cup?"
- Automatic updates: after each action execution, log the event to `TIMELINE.md` and update `MEMORY_SPATIAL.md`

### Out of Scope
- Visual SLAM or point cloud mapping (handled by vision pipeline)
- Hardware driver implementation (Tracks 1-4)
- Task orchestration (Track 5)

## 3. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `OEA/templates/MEMORY_SPATIAL.md` | Create | Spatial memory index template |
| `OEA/templates/TIMELINE.md` | Create | Chronological event log template |
| `OEA/agent/tools/spatial_memory.py` | Create | `SpatialMemoryTool` implementation |
| `OEA/agent/tools/timeline.py` | Create | `TimelineTool` implementation |
| `OEA/agent/loop.py` | Modify | Register new tools |
| `tests/test_spatial_memory.py` | Create | Memory system tests |

## 4. MEMORY_SPATIAL.md Specification

A spatial index of all known entities, their last-known positions, and spatial relationships.

```markdown
# Spatial Memory

Last updated: 2026-03-15 15:30

## Known Objects

| Object | Type | Last Position | Location | Last Seen | Confidence |
|--------|------|--------------|----------|-----------|------------|
| red_apple | fruit | x=5 y=5 z=0 | table | 2026-03-15 15:28 | high |
| blue_cup | container | x=-10 y=3 z=0 | table | 2026-03-15 15:25 | high |
| go2_robot | quadruped | x=120 y=350 z=0 | hallway | 2026-03-15 15:30 | high |

## Spatial Relationships

- red_apple is ON table
- blue_cup is ON table, RIGHT OF red_apple
- table is IN living_room
- go2_robot is IN hallway, NEAR kitchen_door

## Rooms / Zones

| Zone | Bounds | Contains |
|------|--------|----------|
| living_room | x=[0,200] y=[0,400] | table, sofa, red_apple, blue_cup |
| kitchen | x=[200,400] y=[0,400] | counter, sink |
| hallway | x=[100,150] y=[400,600] | go2_robot |
```

## 5. TIMELINE.md Specification

A chronological log of all significant events.

```markdown
# Timeline

## 2026-03-15

### 15:25 — Environment Scan
- Detected blue_cup on table at x=-10 y=3
- Detected red_apple on table at x=5 y=5

### 15:28 — Action Executed
- **Action**: pick_up red_apple
- **Device**: dobot_nova5
- **Result**: Success
- **State Change**: red_apple location: table → held

### 15:29 — Action Executed
- **Action**: put_down red_apple at shelf
- **Device**: dobot_nova5
- **Result**: Success
- **State Change**: red_apple location: held → shelf

### 15:30 — Navigation
- **Action**: move_to kitchen
- **Device**: go2_edu
- **Result**: Arrived at x=120 y=350
```

## 6. Tool Interface

### SpatialMemoryTool

```python
{
    "name": "query_spatial_memory",
    "parameters": {
        "query_type": "string — 'find_object' | 'near' | 'in_zone' | 'relationships'",
        "object_name": "string — optional, target object",
        "zone_name": "string — optional, target zone",
        "radius_cm": "number — optional, search radius"
    }
}
```

### TimelineTool

```python
{
    "name": "query_timeline",
    "parameters": {
        "query_type": "string — 'recent' | 'object_history' | 'time_range'",
        "object_name": "string — optional",
        "since": "string — optional, ISO timestamp",
        "limit": "number — optional, max events to return"
    }
}
```

## 7. Automatic Update Mechanism

After each action execution:
1. The `EmbodiedActionTool` (existing) writes to `ACTION.md`
2. The HAL Watchdog executes and updates `ENVIRONMENT.md`
3. The memory system diffs `ENVIRONMENT.md` changes and:
   - Updates `MEMORY_SPATIAL.md` with new positions
   - Appends an event entry to `TIMELINE.md`

This can be triggered via the Heartbeat mechanism or as a post-action hook.

## 8. Milestones & Acceptance Criteria

### Milestone M1: Protocol Files
- [ ] `OEA/templates/MEMORY_SPATIAL.md` exists with Known Objects table, Spatial Relationships section, and Rooms/Zones table
- [ ] `OEA/templates/TIMELINE.md` exists with date-headed chronological log format
- [ ] Both files auto-synced to workspace via `sync_workspace_templates()`
- [ ] `context.py` `EMBODIED_FILES` includes `MEMORY_SPATIAL.md` and `TIMELINE.md`

### Milestone M2: Tool Unit Tests
- [ ] `SpatialMemoryTool` and `TimelineTool` register without error
- [ ] `query_spatial_memory(query_type="find_object", object_name="apple")` returns correct position dict
- [ ] `query_spatial_memory(query_type="near", object_name="cup", radius_cm=20)` returns objects within 20 cm
- [ ] `query_timeline(query_type="recent", limit=5)` returns last 5 events in chronological order
- [ ] All unit tests pass with mock files (no hardware)

### Milestone M3: Auto-Update on Action
- [ ] After `pick_up apple` executes, `MEMORY_SPATIAL.md` shows `red_apple.location = "held"` within 2 seconds
- [ ] `TIMELINE.md` contains the new event entry (action, device, result, state change)
- [ ] Memory files persist across `nanobot agent` restarts (confirmed by restarting and querying)

### Milestone M4: Spatial Queries via CLI
- [ ] User asks "where was the apple last seen?" → Agent reads `MEMORY_SPATIAL.md` → correct position returned
- [ ] User asks "what happened in the last 10 minutes?" → Agent reads `TIMELINE.md` → correct event list returned
- [ ] Proximity query "what's near the cup?" returns objects within radius

### Milestone M5: Cross-Session Continuity
- [ ] Agent shut down and restarted → `MEMORY_SPATIAL.md` still contains previous object positions
- [ ] `TIMELINE.md` retains history from before restart
- [ ] Long-session query (>100 events in TIMELINE) performs without noticeable slowdown

## 9. Dependencies

- Existing OEA Tool framework
- `ENVIRONMENT.md` (source of spatial updates)
- `ACTION.md` (source of action events)
