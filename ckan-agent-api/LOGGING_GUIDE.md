# Agent Workflow Logging Guide

This document explains how to use the comprehensive logging system that tracks how queries flow through the agent's nodes and edges.

## Overview

The logging system provides detailed visibility into:
- **Node execution** - when nodes start and end
- **State transitions** - what changes between nodes  
- **Routing decisions** - why the graph chose certain paths
- **Interrupts** - when user input is needed
- **Errors** - exceptions that occur during execution

## Enabling Logging

The logging system is automatically initialized when the agent starts. Logs are written to `stdout` with timestamps.

Example output format:
```
[2026-05-21 14:35:22] ckan_registration.workflow - INFO - 🚀 GRAPH STARTED: Thread=abc123def456 | Initial Action=analyze
[2026-05-21 14:35:22] ckan_registration.workflow - INFO - → ENTERING NODE: intake        | Action: analyze     | Thread: abc123def456 | Reason: Parse and normalize incoming request
```

## Log Levels

- **INFO** - Major workflow events (node entry/exit, routing, graph start/end)
- **DEBUG** - Detailed state information and decision rationale  
- **WARNING** - Interrupts (user input needed)
- **ERROR** - Exceptions and errors

## Log Messages

### Graph Lifecycle

```
🚀 GRAPH STARTED: Thread={thread_id} | Initial Action={action}
   Request: {request_summary}
```
Indicates the workflow is starting. Shows thread ID and initial action type.

---

```
✅ GRAPH COMPLETED SUCCESSFULLY: Thread={thread_id} | Status={final_status}
```
Indicates successful completion.

---

```
🛑 GRAPH COMPLETED WITH ERROR: Thread={thread_id} | Status={final_status}
   Error: {error_message}
```
Indicates the workflow failed with an error.

---

### Node Execution

```
→ ENTERING NODE: {node_name} | Action: {action} | Thread: {thread_id} | Reason: {reason}
   State: {state_snapshot}
```
Logged when entering any node. The reason explains why this node is being executed.

Nodes in your workflow:
- `intake` - Parse and normalize incoming request
- `plan` - Validate inputs and determine routing strategy  
- `analyze` - Run analysis operation
- `revise` - Run revision operation
- `dry-run` - Run dry-run operation (preview without applying)
- `approval` - Wait for user approval before applying
- `apply` - Execute the registration
- `show` - Display current status

---

```
← EXITING NODE:  {node_name} | Status: {status} | Action: {action} → Next: {next_node}
   Result: {result_snapshot}
```
Logged when exiting a node. Shows what node will execute next.

---

### Routing Decisions

```
🔀 ROUTING DECISION: {from_node} → {next_node}
   Reason: {decision_reason}
   Details: {decision_details}
```
Logged at conditional edges to explain why a particular path was chosen.

**Common routing reasons:**

From `plan` node:
- `"User has provided necessary inputs for action 'analyze'"` → routes to `analyze`
- `"User has provided necessary inputs for action 'revise'"` → routes to `revise`
- `"Waiting for user to provide clarification on missing inputs"` → loops back to `plan`
- `"Action 'apply' requires review before execution"` → routes to `approval` 

From `approval` node:
- `"User approved registration"` → routes to `apply`
- `"Awaiting user approval"` → loops back to `approval`

---

### Interrupts (User Input Needed)

```
⏸️  INTERRUPT: {interrupt_type}
   Message: {user_facing_message}
   Details: {interrupt_details}
```
Logged when the workflow pauses to request user input. The message explains what's needed.

**Common interrupts:**

- `clarification_required` - Missing data or metadata to proceed
- `ckan_apply_approval_required` - User must approve before registering dataset

When an interrupt occurs, the workflow waits. The user must resume with the required information.

---

### Errors

```
❌ ERROR in {node_name}: {error_message}
   State at error: {state_dict}
```
Logged when an exception occurs in a node.

---

## Reading the Logs

### Example 1: Successful Simple Query

```
🚀 GRAPH STARTED: Thread=abc123 | Initial Action=analyze
   Request: {'has_session': False, 'has_data': True}

→ ENTERING NODE: intake        | Action: analyze     | Thread: abc123 | Reason: Parse and normalize incoming request
   State: {'action': 'analyze', 'status': 'routed', ...}

← EXITING NODE:  intake        | Status: routed       | Action: analyze → Next: plan
   Result: {'action': 'analyze', 'status': 'routed', ...}

🔀 ROUTING DECISION: intake → plan
   Reason: Action 'analyze' is explicit and valid
   Details: {'action': 'analyze'}

→ ENTERING NODE: plan          | Action: analyze     | Thread: abc123 | Reason: Validate inputs and determine routing strategy
   Inputs available: has_data=True, has_session=False, has_message=True, metadata_valid=False

← EXITING NODE:  plan          | Status: ready       | Action: analyze → Next: analyze
   Result: {'action': 'analyze', 'status': 'ready', ...}

🔀 ROUTING DECISION: plan → analyze
   Reason: User has provided necessary inputs for action 'analyze'
   Details: {'action': 'analyze', 'status': 'ready'}

→ ENTERING NODE: analyze       | Action: analyze     | Thread: abc123 | Reason: Execute analyze operation
   ...analyze runs...

← EXITING NODE:  analyze       | Status: analyzing   | Action: analyze → Next: END
   Result: {'action': 'analyze', 'status': 'analyzing', ...}

✅ GRAPH COMPLETED SUCCESSFULLY: Thread=abc123 | Status=analyzing
```

### Example 2: Query with Missing Data (Interrupt)

```
🚀 GRAPH STARTED: Thread=def456 | Initial Action=analyze
   Request: {'has_session': False, 'has_data': False}

→ ENTERING NODE: plan          | Action: analyze     | Thread: def456 | Reason: Validate inputs and determine routing strategy
   Inputs available: has_data=False, has_session=False, has_message=True, metadata_valid=False
   
   Data source found, checking metadata context ❌
   Action 'analyze' requires data input, but none provided

⏸️  INTERRUPT: clarification_required
   Message: To analyze, I need to know where your data is...
   Details: {'action': 'analyze', 'required_fields': ['upload_dir', ...]}

← EXITING NODE:  plan          | Status: awaiting_clarification | Action: analyze → Next: plan
   Result: {'action': 'analyze', 'status': 'awaiting_clarification', ...}

🔀 ROUTING DECISION: plan → plan
   Reason: Waiting for user to provide clarification on missing inputs
   Details: {'awaiting_fields': ['upload_dir', 'upload_dirs', ...]}
```

The workflow is now paused. When the user resumes with the missing data:

```
📋 RESUMING workflow: Thread=def456
   [graph runs again with new data]

✅ GRAPH COMPLETED SUCCESSFULLY: Thread=def456 | Status=analyzing
```

---

## Interpreting the Flow

### State Fields Tracked

The logs show these state fields:
- `action` - The operation to perform (analyze, revise, dry-run, apply, show)
- `status` - Current state (routed, ready, approved, error, etc.)
- `requires_action` - If non-null, indicates special handling needed

### Why Decisions Are Made

Each routing decision logs its reasoning:

| Decision | Reason | Implication |
|----------|--------|-------------|
| plan → plan | "Waiting for clarification..." | Loop back, user must provide missing info |
| plan → analyze | "User has provided necessary inputs..." | Ready to analyze dataset |
| plan → dry-run | "User wants preview before applying..." | Preview changes without applying |
| plan → approval | "User wants to register dataset" | Route to approval before applying |
| approval → apply | "User approved with 'REGISTER'" | Proceed with registration |

---

## Troubleshooting

### "Awaiting clarification" - Why?

Check the interrupt details to see what's missing:
- `upload_dir`, `upload_dirs` - Where's your data file?
- `title_or_name` - Dataset name missing
- `notes_or_description` - Dataset description missing
- `session_id` - Trying to resume/revise but no session found

### Node keeps looping to itself

The plan node may keep interrupting because:
1. Data is found but metadata is missing → add `title` and `notes`  
2. Data source is invalid → check file path or URL
3. Session doesn't exist for `revise`/`show` → use `analyze` instead

### Unexpected routing

Check the routing decision log. It explains why the graph chose that path. Common surprises:
- Graph routes to "plan" → usually awaiting clarification (check interrupts)
- Graph routes to "approval" → you specified `action: "apply"` instead of `dry-run`

---

## Debug Output

For more detailed debug information, look for lines with timestamps prefixed by:
```
... DEBUG ...   Details: {...}
... DEBUG ...   State: {...}
```

These show intermediate state and decision details.

---

## Integration with Your Application

The logs are automatically written to stderr/stdout. To capture them in your application:

```python
import logging

# Get the logger
logger = logging.getLogger("ckan_registration.workflow")

# You can add custom handlers if needed:
handler = logging.FileHandler("workflow.log")
logger.addHandler(handler)
```

Or run the server with:
```bash
uvicorn app.main:app --reload 2>&1 | tee workflow.log
```

This captures both stdout and stderr to a file while also displaying them in the terminal.
