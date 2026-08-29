"""Path Security Policy & Traversal Defense Subsystem.

Provides strict path canonicalization, traversal defense, restricted workspace scoping,
sensitive file pattern rejection (.env, credentials, private keys), and file size enforcement.
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field

# Sensitive filename patterns and extensions strictly forbidden from tool access
RESTRICTED_PATTERNS = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "credentials",
    "credentials.json",
    "secrets.json",
    "jwt.hex",
    "master.key",
    "private.pem",
    "private.key",
}

RESTRICTED_EXTENSIONS = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".kdbx",
}


class PathValidationResult(BaseModel):
    """Path validation outcome returned by PathPolicy."""

    is_valid: bool = Field(description="True if path passes all security policy checks")
    canonical_path: str | None = Field(default=None, description="Resolved absolute path")
    reason: str | None = Field(default=None, description="Detailed rejection reason if invalid")
    violations: list[str] = Field(
        default_factory=list, description="Security policy violation flags"
    )


class PathPolicy:
    """Strict Path Security Policy enforcing directory boundaries and secrets protection."""

    def __init__(
        self,
        allowed_directories: list[str] | None = None,
        max_file_size_bytes: int = 10 * 1024 * 1024,  # 10 MB
    ) -> None:
        """Initializes PathPolicy.

        Args:
            allowed_directories: Optional list of allowed absolute directory paths.
            max_file_size_bytes: Maximum allowed file size limit in bytes.
        """
        if allowed_directories is None:
            # Default to current project workspace directory
            curr_dir = os.path.abspath(os.getcwd())
            self.allowed_directories = [curr_dir]
        else:
            self.allowed_directories = [
                os.path.abspath(os.path.realpath(d)) for d in allowed_directories
            ]

        self.max_file_size_bytes = max_file_size_bytes

    def validate_path(
        self, target_path: str, check_exists: bool = False, max_bytes: int | None = None
    ) -> PathValidationResult:
        """Validates target path against traversal attacks, restricted files, and directory boundaries.

        Args:
            target_path: Input file or directory path string.
            check_exists: If True, asserts that the file/dir must exist on filesystem.
            max_bytes: Custom size limit override.

        Returns:
            PathValidationResult: Path validation outcome.
        """
        violations: list[str] = []

        if not target_path or not target_path.strip():
            return PathValidationResult(
                is_valid=False,
                reason="Path validation failed: Target path string is empty.",
                violations=["EMPTY_PATH"],
            )

        raw_path_str = target_path.strip()

        # Step 1: Traversal pattern detection
        if ".." in raw_path_str or raw_path_str.startswith("~"):
            # Check canonicalization
            pass

        try:
            # Canonicalize path (resolving symlinks and relative segments)
            p = Path(raw_path_str)
            resolved_path = os.path.abspath(os.path.realpath(str(p)))
        except Exception as exc:
            return PathValidationResult(
                is_valid=False,
                reason=f"Path validation failed: Invalid path format: {exc}",
                violations=["INVALID_FORMAT"],
            )

        # Check traversal escape
        basename = os.path.basename(resolved_path).lower()

        # Step 2: Sensitive file and secret pattern checks
        if basename in RESTRICTED_PATTERNS or any(
            basename.startswith(pat) for pat in [".env", "id_rsa", "id_ed25519"]
        ):
            violations.append("RESTRICTED_SENSITIVE_FILE")
            return PathValidationResult(
                is_valid=False,
                canonical_path=resolved_path,
                reason=f"Security Policy Violation: Access to sensitive file '{basename}' is strictly forbidden.",
                violations=violations,
            )

        ext = os.path.splitext(basename)[1].lower()
        if ext in RESTRICTED_EXTENSIONS:
            violations.append("RESTRICTED_FILE_EXTENSION")
            return PathValidationResult(
                is_valid=False,
                canonical_path=resolved_path,
                reason=f"Security Policy Violation: Access to key/credential extension '{ext}' is strictly forbidden.",
                violations=violations,
            )

        # Step 3: Allowed Directory Boundary Checking
        in_allowed_dir = False
        for allowed_dir in self.allowed_directories:
            if resolved_path == allowed_dir or resolved_path.startswith(allowed_dir + os.sep):
                in_allowed_dir = True
                break

        if not in_allowed_dir:
            violations.append("OUTSIDE_ALLOWED_DIRECTORY")
            return PathValidationResult(
                is_valid=False,
                canonical_path=resolved_path,
                reason=(
                    f"Security Policy Violation: Target path '{resolved_path}' is outside "
                    f"allowed workspace directories {self.allowed_directories}."
                ),
                violations=violations,
            )

        # Step 4: Existence & File Size Checking (if required)
        if check_exists:
            if not os.path.exists(resolved_path):
                violations.append("FILE_NOT_FOUND")
                return PathValidationResult(
                    is_valid=False,
                    canonical_path=resolved_path,
                    reason=f"Path validation failed: File or directory '{resolved_path}' does not exist.",
                    violations=violations,
                )

            if os.path.isfile(resolved_path):
                size_limit = max_bytes or self.max_file_size_bytes
                actual_size = os.path.getsize(resolved_path)
                if actual_size > size_limit:
                    violations.append("FILE_SIZE_EXCEEDED")
                    return PathValidationResult(
                        is_valid=False,
                        canonical_path=resolved_path,
                        reason=(
                            f"Security Policy Violation: File size ({actual_size} bytes) "
                            f"exceeds maximum allowed limit of {size_limit} bytes."
                        ),
                        violations=violations,
                    )

        return PathValidationResult(
            is_valid=True,
            canonical_path=resolved_path,
            reason="Path validated successfully under security policy.",
            violations=[],
        )


# Global PathPolicy singleton
path_policy = PathPolicy()
