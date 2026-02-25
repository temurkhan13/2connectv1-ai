"""
Smart Edit Service with Dependency Graph.

Handles cascading updates when users edit their responses.
Uses a dependency graph to track which slots depend on others
and propagates invalidation appropriately.

Key features:
1. Dependency graph construction and traversal
2. Cascading invalidation of dependent slots
3. Edit impact analysis before applying changes
4. Undo/redo support for edits
5. Conflict detection and resolution
"""
import os
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum

from app.services.slot_extraction import SlotDefinition, SlotStatus, SlotSchema
from app.services.context_manager import (
    ContextManager, ConversationContext, ExtractedSlot
)

logger = logging.getLogger(__name__)


class EditType(str, Enum):
    """Type of edit operation."""
    UPDATE = "update"      # Change value
    CLEAR = "clear"        # Remove value
    CONFIRM = "confirm"    # Confirm value
    SKIP = "skip"          # Skip slot


class ImpactLevel(str, Enum):
    """Impact level of an edit."""
    NONE = "none"          # No cascading impact
    LOW = "low"            # 1-2 dependent slots
    MEDIUM = "medium"      # 3-5 dependent slots
    HIGH = "high"          # 6+ or critical slots affected


@dataclass
class EditOperation:
    """Represents a single edit operation."""
    edit_id: str
    slot_name: str
    edit_type: EditType
    old_value: Any
    new_value: Any
    timestamp: datetime
    affected_slots: List[str] = field(default_factory=list)
    applied: bool = False


@dataclass
class EditImpactAnalysis:
    """Analysis of edit impact before applying."""
    slot_name: str
    edit_type: EditType
    new_value: Any
    impact_level: ImpactLevel
    affected_slots: List[str]
    will_invalidate: List[str]
    requires_re_prompt: List[str]
    warnings: List[str]
    can_proceed: bool = True


@dataclass
class DependencyNode:
    """Node in the dependency graph."""
    slot_name: str
    depends_on: Set[str] = field(default_factory=set)
    depended_by: Set[str] = field(default_factory=set)
    is_critical: bool = False


class DependencyGraph:
    """
    Tracks slot dependencies for cascading updates.

    Supports bidirectional traversal to find both
    upstream dependencies and downstream dependents.
    """

    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """Build dependency graph from slot schema."""
        schema = SlotSchema()

        # Collect all slot definitions (convert lists to dict keyed by name)
        all_slots = {}
        for slot_list in [schema.CORE_SLOTS, schema.INVESTOR_SLOTS,
                          schema.FOUNDER_SLOTS, schema.OPTIONAL_SLOTS]:
            for slot_def in slot_list:
                all_slots[slot_def.name] = slot_def

        # Create nodes
        for slot_name, slot_def in all_slots.items():
            node = DependencyNode(
                slot_name=slot_name,
                is_critical=slot_def.required
            )

            # Add dependencies
            if slot_def.depends_on:
                node.depends_on = set(slot_def.depends_on)

            self.nodes[slot_name] = node

        # Build reverse dependencies (depended_by)
        for slot_name, node in self.nodes.items():
            for dep in node.depends_on:
                if dep in self.nodes:
                    self.nodes[dep].depended_by.add(slot_name)

        # Add implicit dependencies
        self._add_implicit_dependencies()

    def _add_implicit_dependencies(self) -> None:
        """Add implicit dependencies not in schema."""
        # user_type determines which role-specific slots are relevant
        if "user_type" in self.nodes:
            investor_slots = ["check_size", "portfolio_size", "investment_thesis"]
            founder_slots = ["company_stage", "funding_need", "team_size"]

            for slot in investor_slots + founder_slots:
                if slot in self.nodes:
                    self.nodes[slot].depends_on.add("user_type")
                    self.nodes["user_type"].depended_by.add(slot)

    def get_dependents(self, slot_name: str, recursive: bool = True) -> Set[str]:
        """
        Get all slots that depend on the given slot.

        Args:
            slot_name: Slot to check
            recursive: Whether to include transitive dependents

        Returns:
            Set of dependent slot names
        """
        if slot_name not in self.nodes:
            return set()

        dependents = set(self.nodes[slot_name].depended_by)

        if recursive:
            to_check = list(dependents)
            while to_check:
                current = to_check.pop()
                if current in self.nodes:
                    new_deps = self.nodes[current].depended_by - dependents
                    dependents.update(new_deps)
                    to_check.extend(new_deps)

        return dependents

    def get_dependencies(self, slot_name: str, recursive: bool = True) -> Set[str]:
        """
        Get all slots that the given slot depends on.

        Args:
            slot_name: Slot to check
            recursive: Whether to include transitive dependencies

        Returns:
            Set of dependency slot names
        """
        if slot_name not in self.nodes:
            return set()

        dependencies = set(self.nodes[slot_name].depends_on)

        if recursive:
            to_check = list(dependencies)
            while to_check:
                current = to_check.pop()
                if current in self.nodes:
                    new_deps = self.nodes[current].depends_on - dependencies
                    dependencies.update(new_deps)
                    to_check.extend(new_deps)

        return dependencies

    def get_critical_path(self, slot_name: str) -> List[str]:
        """
        Get the critical path of dependencies for a slot.

        Returns slots in order they should be filled.
        """
        dependencies = self.get_dependencies(slot_name)

        # Topological sort
        result = []
        visited = set()

        def visit(node: str):
            if node in visited:
                return
            visited.add(node)
            if node in self.nodes:
                for dep in self.nodes[node].depends_on:
                    visit(dep)
            result.append(node)

        for dep in dependencies:
            visit(dep)

        return result

    def has_circular_dependency(self, slot_name: str) -> bool:
        """Check if slot has circular dependencies."""
        visited = set()
        path = set()

        def dfs(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False

            visited.add(node)
            path.add(node)

            if node in self.nodes:
                for dep in self.nodes[node].depends_on:
                    if dfs(dep):
                        return True

            path.remove(node)
            return False

        return dfs(slot_name)


class SmartEdit:
    """
    Manages smart editing of slot values with cascading updates.

    Provides impact analysis before edits and handles
    propagation of changes through the dependency graph.
    """

    def __init__(self, context_manager: Optional[ContextManager] = None):
        self.context_manager = context_manager or ContextManager()
        self.graph = DependencyGraph()

        # Edit history per session for undo/redo
        self._edit_history: Dict[str, List[EditOperation]] = defaultdict(list)
        self._redo_stack: Dict[str, List[EditOperation]] = defaultdict(list)

        # Configuration
        self.max_history_size = int(os.getenv("EDIT_HISTORY_SIZE", "20"))

    def analyze_edit(
        self,
        session_id: str,
        slot_name: str,
        new_value: Any,
        edit_type: EditType = EditType.UPDATE
    ) -> EditImpactAnalysis:
        """
        Analyze the impact of an edit before applying it.

        Args:
            session_id: Session identifier
            slot_name: Slot to edit
            new_value: New value to set
            edit_type: Type of edit

        Returns:
            EditImpactAnalysis with impact details
        """
        context = self.context_manager.get_session(session_id)
        if not context:
            return EditImpactAnalysis(
                slot_name=slot_name,
                edit_type=edit_type,
                new_value=new_value,
                impact_level=ImpactLevel.NONE,
                affected_slots=[],
                will_invalidate=[],
                requires_re_prompt=[],
                warnings=["Session not found"],
                can_proceed=False
            )

        # Get all dependent slots
        dependents = self.graph.get_dependents(slot_name)
        affected = list(dependents)

        # Determine which filled slots would be invalidated
        will_invalidate = []
        requires_re_prompt = []

        for dep_slot in dependents:
            existing = context.slots.get(dep_slot)
            if existing and existing.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                will_invalidate.append(dep_slot)
                requires_re_prompt.append(dep_slot)

        # Calculate impact level
        if len(will_invalidate) == 0:
            impact_level = ImpactLevel.NONE
        elif len(will_invalidate) <= 2:
            impact_level = ImpactLevel.LOW
        elif len(will_invalidate) <= 5:
            impact_level = ImpactLevel.MEDIUM
        else:
            impact_level = ImpactLevel.HIGH

        # Generate warnings
        warnings = []

        # Check for critical slot edits
        if slot_name in self.graph.nodes and self.graph.nodes[slot_name].is_critical:
            warnings.append(f"'{slot_name}' is a critical slot that affects matching.")

        # Check for major impact
        if impact_level == ImpactLevel.HIGH:
            warnings.append(
                f"This edit will invalidate {len(will_invalidate)} other responses."
            )

        # Special warning for user_type change
        if slot_name == "user_type":
            warnings.append(
                "Changing user type will reset all role-specific questions."
            )

        # Check for circular dependencies
        if self.graph.has_circular_dependency(slot_name):
            warnings.append("Warning: Circular dependency detected.")

        return EditImpactAnalysis(
            slot_name=slot_name,
            edit_type=edit_type,
            new_value=new_value,
            impact_level=impact_level,
            affected_slots=affected,
            will_invalidate=will_invalidate,
            requires_re_prompt=requires_re_prompt,
            warnings=warnings,
            can_proceed=True
        )

    def apply_edit(
        self,
        session_id: str,
        slot_name: str,
        new_value: Any,
        edit_type: EditType = EditType.UPDATE,
        force: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply an edit and cascade updates.

        Args:
            session_id: Session identifier
            slot_name: Slot to edit
            new_value: New value to set
            edit_type: Type of edit
            force: Apply without confirmation for high-impact edits

        Returns:
            Tuple of (success, result_details)
        """
        context = self.context_manager.get_session(session_id)
        if not context:
            return False, {"error": "Session not found"}

        # Analyze impact first
        analysis = self.analyze_edit(session_id, slot_name, new_value, edit_type)

        if not analysis.can_proceed:
            return False, {"error": "Edit cannot proceed", "warnings": analysis.warnings}

        # Require confirmation for high impact edits
        if analysis.impact_level == ImpactLevel.HIGH and not force:
            return False, {
                "error": "High impact edit requires confirmation",
                "analysis": {
                    "impact_level": analysis.impact_level.value,
                    "will_invalidate": analysis.will_invalidate,
                    "warnings": analysis.warnings
                }
            }

        # Get old value for history
        old_slot = context.slots.get(slot_name)
        old_value = old_slot.value if old_slot else None

        # Create edit operation
        edit_op = EditOperation(
            edit_id=f"{session_id}_{datetime.utcnow().timestamp()}",
            slot_name=slot_name,
            edit_type=edit_type,
            old_value=old_value,
            new_value=new_value,
            timestamp=datetime.utcnow(),
            affected_slots=analysis.affected_slots
        )

        # Apply the edit
        try:
            if edit_type == EditType.UPDATE:
                self._apply_update(context, slot_name, new_value)
            elif edit_type == EditType.CLEAR:
                self._apply_clear(context, slot_name)
            elif edit_type == EditType.CONFIRM:
                self._apply_confirm(context, slot_name)
            elif edit_type == EditType.SKIP:
                self._apply_skip(context, slot_name)

            # Cascade invalidation
            invalidated = self._cascade_invalidation(context, analysis.will_invalidate)

            # Record in history
            edit_op.applied = True
            self._add_to_history(session_id, edit_op)

            # Clear redo stack on new edit
            self._redo_stack[session_id] = []

            result = {
                "success": True,
                "slot_name": slot_name,
                "new_value": new_value,
                "invalidated_slots": invalidated,
                "requires_re_prompt": analysis.requires_re_prompt
            }

            logger.info(
                f"Applied edit to {slot_name}: {old_value} -> {new_value}, "
                f"invalidated {len(invalidated)} slots"
            )

            return True, result

        except Exception as e:
            logger.error(f"Error applying edit: {e}")
            return False, {"error": str(e)}

    def _apply_update(
        self,
        context: ConversationContext,
        slot_name: str,
        new_value: Any
    ) -> None:
        """Apply an update edit."""
        if slot_name in context.slots:
            context.slots[slot_name].value = new_value
            context.slots[slot_name].status = SlotStatus.FILLED
            context.slots[slot_name].confidence = 1.0  # User-confirmed
        else:
            context.slots[slot_name] = ExtractedSlot(
                slot_name=slot_name,
                value=new_value,
                confidence=1.0,
                status=SlotStatus.FILLED,
                source_text="user_edit"
            )
        context.updated_at = datetime.utcnow()

    def _apply_clear(self, context: ConversationContext, slot_name: str) -> None:
        """Apply a clear edit."""
        if slot_name in context.slots:
            context.slots[slot_name].value = None
            context.slots[slot_name].status = SlotStatus.EMPTY
        context.updated_at = datetime.utcnow()

    def _apply_confirm(self, context: ConversationContext, slot_name: str) -> None:
        """Apply a confirm edit."""
        if slot_name in context.slots:
            context.slots[slot_name].status = SlotStatus.CONFIRMED
        context.updated_at = datetime.utcnow()

    def _apply_skip(self, context: ConversationContext, slot_name: str) -> None:
        """Apply a skip edit."""
        context.slots[slot_name] = ExtractedSlot(
            slot_name=slot_name,
            value=None,
            confidence=1.0,
            status=SlotStatus.SKIPPED,
            source_text="user_skip"
        )
        context.updated_at = datetime.utcnow()

    def _cascade_invalidation(
        self,
        context: ConversationContext,
        slots_to_invalidate: List[str]
    ) -> List[str]:
        """Invalidate dependent slots."""
        invalidated = []

        for slot_name in slots_to_invalidate:
            if slot_name in context.slots:
                # Mark as needing re-prompt (keep old value as suggestion)
                context.slots[slot_name].status = SlotStatus.PARTIAL
                invalidated.append(slot_name)

        return invalidated

    def _add_to_history(self, session_id: str, edit_op: EditOperation) -> None:
        """Add edit to history, maintaining size limit."""
        history = self._edit_history[session_id]
        history.append(edit_op)

        if len(history) > self.max_history_size:
            self._edit_history[session_id] = history[-self.max_history_size:]

    def undo(self, session_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Undo the last edit.

        Args:
            session_id: Session identifier

        Returns:
            Tuple of (success, result_details)
        """
        history = self._edit_history.get(session_id, [])
        if not history:
            return False, {"error": "No edits to undo"}

        context = self.context_manager.get_session(session_id)
        if not context:
            return False, {"error": "Session not found"}

        # Pop last edit
        last_edit = history.pop()

        # Restore old value
        if last_edit.old_value is not None:
            self._apply_update(context, last_edit.slot_name, last_edit.old_value)
        else:
            self._apply_clear(context, last_edit.slot_name)

        # Add to redo stack
        self._redo_stack[session_id].append(last_edit)

        logger.info(f"Undid edit to {last_edit.slot_name}")

        return True, {
            "undone": last_edit.slot_name,
            "restored_value": last_edit.old_value
        }

    def redo(self, session_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Redo a previously undone edit.

        Args:
            session_id: Session identifier

        Returns:
            Tuple of (success, result_details)
        """
        redo_stack = self._redo_stack.get(session_id, [])
        if not redo_stack:
            return False, {"error": "No edits to redo"}

        context = self.context_manager.get_session(session_id)
        if not context:
            return False, {"error": "Session not found"}

        # Pop from redo stack
        edit_to_redo = redo_stack.pop()

        # Re-apply the edit
        self._apply_update(context, edit_to_redo.slot_name, edit_to_redo.new_value)

        # Add back to history
        self._edit_history[session_id].append(edit_to_redo)

        logger.info(f"Redid edit to {edit_to_redo.slot_name}")

        return True, {
            "redone": edit_to_redo.slot_name,
            "new_value": edit_to_redo.new_value
        }

    def get_edit_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get edit history for session."""
        history = self._edit_history.get(session_id, [])
        return [
            {
                "edit_id": edit.edit_id,
                "slot_name": edit.slot_name,
                "edit_type": edit.edit_type.value,
                "old_value": edit.old_value,
                "new_value": edit.new_value,
                "timestamp": edit.timestamp.isoformat(),
                "affected_slots": edit.affected_slots
            }
            for edit in history
        ]

    def get_editable_slots(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get list of slots that can be edited.

        Args:
            session_id: Session identifier

        Returns:
            List of editable slot info
        """
        context = self.context_manager.get_session(session_id)
        if not context:
            return []

        editable = []
        for slot_name, slot in context.slots.items():
            # All filled/confirmed slots are editable
            if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED, SlotStatus.PARTIAL]:
                node = self.graph.nodes.get(slot_name)
                editable.append({
                    "slot_name": slot_name,
                    "current_value": slot.value,
                    "status": slot.status.value,
                    "is_critical": node.is_critical if node else False,
                    "dependent_count": len(self.graph.get_dependents(slot_name))
                })

        return editable


# Global instance
smart_edit = SmartEdit()
