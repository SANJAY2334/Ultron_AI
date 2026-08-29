"""Safety Interlock Subsystem.

Evaluates DesktopAction payloads against capability permissions, action classifications
(READ_ONLY, NON_DESTRUCTIVE, MUTATING, DESTRUCTIVE), user confirmation state, and security boundaries.
Returns canonical SafetyDecision (ALLOW, DENY, REQUIRES_CONFIRMATION).
"""

from app.ai.tools.base import ExecutionContext
from app.desktop.models import ActionClassification, DesktopAction, SafetyDecision
from app.security.path_policy import PathPolicy, path_policy


class SafetyInterlock:
    """Safety Interlock evaluating desktop automation actions for multi-stage safety gating."""

    def __init__(self, path_pol: PathPolicy = path_policy) -> None:
        """Initializes SafetyInterlock.

        Args:
            path_pol: PathPolicy instance for filesystem target validation.
        """
        self._path_policy = path_pol

    def evaluate(self, action: DesktopAction, context: ExecutionContext) -> SafetyDecision:
        """Evaluates a DesktopAction against safety interlock rules.

        Args:
            action: Validated DesktopAction domain object.
            context: Active ExecutionContext carrying permissions and environment flags.

        Returns:
            SafetyDecision: ALLOW, DENY, or REQUIRES_CONFIRMATION.
        """
        # Step 1: Capability Permission Check
        if action.capability not in context.granted_capabilities:
            return SafetyDecision(
                decision="DENY",
                reason=(
                    f"Safety Interlock Denied: Action '{action.action_type}' requires capability "
                    f"'{action.capability}' which is not granted in ExecutionContext."
                ),
                missing_capabilities=[action.capability],
            )

        # Step 2: Target Validation (if target looks like a file path)
        if action.capability.startswith("file:"):
            check_exists = action.classification in (
                ActionClassification.READ_ONLY,
                ActionClassification.DESTRUCTIVE,
            )
            path_res = self._path_policy.validate_path(action.target, check_exists=check_exists)
            if not path_res.is_valid:
                return SafetyDecision(
                    decision="DENY",
                    reason=f"Safety Interlock Denied: Path security violation: {path_res.reason}",
                    path_violations=path_res.violations,
                )

        # Step 3: Action Classification Rules
        classification = action.classification

        # Rule 1: READ_ONLY -> Automatically allowed when capability granted
        if classification == ActionClassification.READ_ONLY:
            return SafetyDecision(
                decision="ALLOW",
                reason=f"READ_ONLY action '{action.action_type}' authorized by SafetyInterlock.",
            )

        # Rule 2: NON_DESTRUCTIVE -> Allowed when capability granted
        if classification == ActionClassification.NON_DESTRUCTIVE:
            return SafetyDecision(
                decision="ALLOW",
                reason=f"NON_DESTRUCTIVE action '{action.action_type}' authorized by SafetyInterlock.",
            )

        # Rule 3: MUTATING -> Allowed if explicit confirmation provided or not confirmation_required
        if classification == ActionClassification.MUTATING:
            if action.confirmation_required:
                user_confirmed = context.environment.get("user_confirmed", False)
                if not user_confirmed:
                    return SafetyDecision(
                        decision="REQUIRES_CONFIRMATION",
                        reason=(
                            f"Confirmation Required: MUTATING action '{action.action_type}' "
                            f"on target '{action.target}' requires explicit user approval."
                        ),
                    )
            return SafetyDecision(
                decision="ALLOW",
                reason=f"MUTATING action '{action.action_type}' authorized by SafetyInterlock.",
            )

        # Rule 4: DESTRUCTIVE -> MANDATORY Explicit User Confirmation required
        if classification == ActionClassification.DESTRUCTIVE:
            user_confirmed = context.environment.get("user_confirmed", False)
            if not user_confirmed:
                return SafetyDecision(
                    decision="REQUIRES_CONFIRMATION",
                    reason=(
                        f"Explicit Confirmation Required: DESTRUCTIVE action '{action.action_type}' "
                        f"on target '{action.target}' cannot execute without explicit human approval."
                    ),
                )

            return SafetyDecision(
                decision="ALLOW",
                reason=f"DESTRUCTIVE action '{action.action_type}' authorized with user confirmation.",
            )

        return SafetyDecision(
            decision="DENY",
            reason=f"Safety Interlock Denied: Unknown classification '{classification}'.",
        )


# Global SafetyInterlock singleton
safety_interlock = SafetyInterlock()
