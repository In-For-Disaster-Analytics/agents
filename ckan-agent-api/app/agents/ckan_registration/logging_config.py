"""Logging configuration for agent workflow tracking."""

import logging
import json
from datetime import datetime
from typing import Any, Optional

# Create logger for agent workflow
logger = logging.getLogger("ckan_registration.workflow")


def setup_workflow_logging() -> logging.Logger:
    """Configure workflow logger with detailed formatting."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def log_node_entry(node_name: str, state: dict[str, Any], reason: str = "") -> None:
    """Log when entering a node with context."""
    action = state.get("action", "unknown")
    thread_id = state.get("thread_id", "unknown")
    reason_str = f" | Reason: {reason}" if reason else ""
    logger.info(
        f"→ ENTERING NODE: {node_name:12} | Action: {action:10} | Thread: {thread_id}{reason_str}"
    )
    _log_state_snapshot(state, indent="  ")


def log_node_exit(node_name: str, result: dict[str, Any], next_node: Optional[str] = None) -> None:
    """Log when exiting a node with result."""
    action = result.get("action", "unchanged")
    status = result.get("status", "unknown")
    next_str = f" → Next: {next_node}" if next_node else ""
    logger.info(
        f"← EXITING NODE:  {node_name:12} | Status: {status:20} | Action: {action:10}{next_str}"
    )
    _log_result_snapshot(result, indent="  ")


def log_routing_decision(from_node: str, decision_reason: str, next_node: str, details: dict[str, Any]) -> None:
    """Log routing decision at a conditional edge."""
    logger.info(
        f"🔀 ROUTING DECISION: {from_node} → {next_node}"
    )
    logger.info(f"   Reason: {decision_reason}")
    if details:
        logger.debug(f"   Details: {json.dumps(details, default=str)}")


def log_interrupt(interrupt_type: str, message: str, details: dict[str, Any]) -> None:
    """Log when an interrupt occurs (awaiting user input)."""
    logger.warning(
        f"⏸️  INTERRUPT: {interrupt_type}"
    )
    logger.warning(f"   Message: {message}")
    if details:
        logger.debug(f"   Details: {json.dumps(details, default=str)}")


def log_error(node_name: str, error: str, state: dict[str, Any]) -> None:
    """Log errors in node execution."""
    logger.error(f"❌ ERROR in {node_name}: {error}")
    logger.debug(f"   State at error: {json.dumps(_safe_serialize(state), indent=2, default=str)}")


def log_state_transition(from_state: dict[str, Any], to_state: dict[str, Any]) -> None:
    """Log state transitions showing what changed."""
    logger.debug("📊 STATE TRANSITION:")
    _log_state_diff(from_state, to_state)


def log_graph_start(thread_id: str, action: str, request_summary: dict[str, Any]) -> None:
    """Log graph invocation start."""
    logger.info(f"🚀 GRAPH STARTED: Thread={thread_id} | Initial Action={action}")
    logger.debug(f"   Request: {json.dumps(request_summary, default=str)}")


def log_graph_end(thread_id: str, final_status: str, error: Optional[str] = None) -> None:
    """Log graph completion."""
    if error:
        logger.error(f"🛑 GRAPH COMPLETED WITH ERROR: Thread={thread_id} | Status={final_status}")
        logger.error(f"   Error: {error}")
    else:
        logger.info(f"✅ GRAPH COMPLETED SUCCESSFULLY: Thread={thread_id} | Status={final_status}")


def _log_state_snapshot(state: dict[str, Any], indent: str = "") -> None:
    """Log relevant state fields."""
    relevant_keys = ["action", "status", "thread_id", "requires_action"]
    snapshot = {k: v for k, v in state.items() if k in relevant_keys}
    if snapshot:
        logger.debug(f"{indent}State: {json.dumps(snapshot, default=str)}")


def _log_result_snapshot(result: dict[str, Any], indent: str = "") -> None:
    """Log relevant result fields."""
    relevant_keys = ["action", "status", "requires_action", "error"]
    snapshot = {k: v for k, v in result.items() if k in relevant_keys}
    if snapshot:
        logger.debug(f"{indent}Result: {json.dumps(snapshot, default=str)}")


def _log_state_diff(from_state: dict[str, Any], to_state: dict[str, Any]) -> None:
    """Log what changed between states."""
    from_relevant = {k: v for k, v in from_state.items() if k in ["action", "status", "requires_action"]}
    to_relevant = {k: v for k, v in to_state.items() if k in ["action", "status", "requires_action"]}
    
    if from_relevant != to_relevant:
        for key in from_relevant:
            if key not in to_relevant or from_relevant[key] != to_relevant.get(key):
                logger.debug(f"   {key}: {from_relevant.get(key)} → {to_relevant.get(key)}")
        for key in to_relevant:
            if key not in from_relevant:
                logger.debug(f"   {key}: (new) → {to_relevant[key]}")


def _safe_serialize(obj: Any) -> Any:
    """Safely serialize objects for logging."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


# Initialize logger on import
setup_workflow_logging()
# Test that logger is working
logger.info("✅ Workflow logger initialized and ready")