"""
arc_lint.py — Static analysis and code quality tool for the ARC-AGI project.

Usage:
    python arc_lint.py [file1.py file2.py ...]
    python arc_lint.py              # defaults to ArcAgent.py ArcDriver.py ArcProblem.py

Pipeline:
    AST analysis → quality/AI-slop heuristics → human-readable report →
    optional targeted refactoring → run ArcDriver.py → compare score

Behavior preservation is the highest priority. Nothing is rewritten automatically.
"""

import ast
import os
import sys
import textwrap
import tokenize
import io
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Issue dataclass
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    severity: str        # HIGH / MEDIUM / LOW
    filename: str
    line: int
    code: str            # short identifier, e.g. LONG_FUNC
    message: str
    suggestion: str = ""

    def __str__(self):
        sug = f"\n        Suggestion: {self.suggestion}" if self.suggestion else ""
        return f"[{self.severity}] {self.filename}:{self.line}  ({self.code})\n        {self.message}{sug}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESTATE_PATTERNS = [
    r"^#\s*(return|returns|returns the|get|gets|set|sets|check|checks|find|finds|"
    r"loop|loops|iterate|iterates|iterate over|for each|append|add|adds|remove|removes|"
    r"call|calls|compute|computes|calculate|calculates|create|creates|initialize|initializes|"
    r"print|log|raise|catch|try|if|else)\b",
]

GENERIC_COMMENT_PATTERNS = [
    r"^#\s*(this function|this method|this class|this module|helper function|"
    r"utility function|processes the data|handles the|main logic|"
    r"implementation|do the thing|does the thing|TODO|FIXME|HACK|XXX)\b",
    r"^#\s*-{4,}$",   # long separator lines used as decoration
]

ENTERPRISE_PREFIXES = [
    "Manager", "Handler", "Processor", "Controller", "Factory",
    "Builder", "Orchestrator", "Coordinator", "Dispatcher", "Executor",
    "Aggregator", "Transformer", "Resolver", "Provider", "Adapter",
]

MAGIC_NUMBER_WHITELIST = {0, 1, 2, -1, 100, 255}

KNOWN_NUMPY_FUNCS = {"np", "numpy"}


def _line_text(lines, lineno):
    """Return source line text (1-indexed), stripped."""
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _src_lines(source):
    return source.splitlines()


# ---------------------------------------------------------------------------
# AST-based checks
# ---------------------------------------------------------------------------

class ArcLinter(ast.NodeVisitor):

    def __init__(self, filename, source):
        self.filename = filename
        self.source = source
        self.lines = _src_lines(source)
        self.issues = []
        self._func_stack = []       # stack of (name, lineno) for nesting
        self._imports = {}          # name → lineno
        self._used_names = set()
        self._func_defs = {}        # name → lineno
        self._func_bodies = {}      # name → source lines list

    def _issue(self, severity, line, code, message, suggestion=""):
        self.issues.append(Issue(severity, self.filename, line, code, message, suggestion))

    # ------------------------------------------------------------------
    # Visitor methods
    # ------------------------------------------------------------------

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self._imports[name] = node.lineno
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self._imports[name] = node.lineno
        self.generic_visit(node)

    def visit_Name(self, node):
        self._used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self._used_names.add(node.value.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._check_function(node)
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        # Enterprise abstraction names
        for prefix in ENTERPRISE_PREFIXES:
            if node.name.endswith(prefix) and node.name != prefix:
                self._issue("LOW", node.lineno, "ENTERPRISE_NAME",
                            f"Class '{node.name}' uses an enterprise-sounding suffix '{prefix}'.",
                            "Use a plain noun that describes what the class is, not what it does.")
        self.generic_visit(node)

    def visit_Try(self, node):
        self._check_bare_except(node)
        self._check_redundant_try(node)
        self.generic_visit(node)

    def visit_Constant(self, node):
        # Magic numbers (not in whitelist, inside a function, not in an assignment target)
        if isinstance(node.n if hasattr(node, 'n') else None, (int, float)):
            val = node.n
            if val not in MAGIC_NUMBER_WHITELIST and not isinstance(val, bool):
                # Only flag numbers that appear as standalone values in comparisons/slices
                pass   # checked in _check_magic_numbers via token walk
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Function checks
    # ------------------------------------------------------------------

    def _check_function(self, node):
        name = node.name
        lineno = node.lineno
        self._func_defs[name] = lineno

        # Collect body source for duplicate detection
        body_lines = self.lines[node.lineno - 1: node.end_lineno]
        self._func_bodies[name] = body_lines

        # 1. Function length
        func_len = node.end_lineno - node.lineno + 1
        if func_len > 60:
            self._issue("HIGH", lineno, "LONG_FUNC",
                        f"Function '{name}' is {func_len} lines long.",
                        "Split into focused sub-functions of ≤30 lines each.")
        elif func_len > 40:
            self._issue("MEDIUM", lineno, "LONG_FUNC",
                        f"Function '{name}' is {func_len} lines long.",
                        "Consider splitting into smaller functions.")

        # 2. Too many parameters
        args = node.args
        n_args = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
        if n_args > 6:
            self._issue("MEDIUM", lineno, "TOO_MANY_PARAMS",
                        f"Function '{name}' has {n_args} parameters.",
                        "Consider grouping related parameters into a dataclass or dict.")

        # 3. Unused parameters
        param_names = {a.arg for a in args.args + args.posonlyargs + args.kwonlyargs}
        param_names.discard("self")
        param_names.discard("cls")
        body_visitor = NameCollector()
        for stmt in node.body:
            body_visitor.visit(stmt)
        used_in_body = body_visitor.names
        for param in param_names:
            if param not in used_in_body:
                self._issue("MEDIUM", lineno, "UNUSED_PARAM",
                            f"Parameter '{param}' in '{name}' is never used.",
                            "Remove it or prefix with '_' if intentionally unused.")

        # 4. Docstring quality
        if (ast.get_docstring(node)):
            docstring = ast.get_docstring(node)
            self._check_docstring_quality(docstring, name, lineno)

        # 5. Nesting depth
        max_depth = _max_nesting_depth(node)
        if max_depth >= 5:
            self._issue("HIGH", lineno, "DEEP_NESTING",
                        f"Function '{name}' has nesting depth {max_depth}.",
                        "Flatten with early returns or extract inner logic to helper functions.")
        elif max_depth >= 4:
            self._issue("MEDIUM", lineno, "DEEP_NESTING",
                        f"Function '{name}' has nesting depth {max_depth}.",
                        "Consider early returns to reduce nesting.")

        # 6. Overly verbose name
        words = _split_name(name)
        if len(words) >= 6:
            self._issue("LOW", lineno, "VERBOSE_NAME",
                        f"Function name '{name}' has {len(words)} words.",
                        "Shorten: names should communicate intent in 2–4 words.")

        # 7. Name is too generic
        GENERIC_FUNC_NAMES = {"process", "handle", "do_thing", "run_logic",
                               "execute", "perform", "manage", "process_data",
                               "handle_data", "do_stuff"}
        if name.lower() in GENERIC_FUNC_NAMES:
            self._issue("LOW", lineno, "GENERIC_NAME",
                        f"Function name '{name}' is too generic.",
                        "Use a name that describes the specific transformation or operation.")

        # 8. Returns inside try that could be simplified
        # (checked in visit_Try)

    def _check_docstring_quality(self, docstring, func_name, lineno):
        doc = docstring.strip().lower()

        # Generic docstring
        generic_phrases = [
            "this function", "this method", "processes", "handles",
            "performs the", "does the", "main logic", "helper"
        ]
        for phrase in generic_phrases:
            if phrase in doc and len(doc) < 80:
                self._issue("LOW", lineno, "GENERIC_DOCSTRING",
                            f"Docstring for '{func_name}' is generic: '{docstring[:60].strip()}'",
                            "Write what the function specifically does, not that it 'processes' something.")
                break

        # Docstring that just restates the function name
        name_words = set(_split_name(func_name))
        doc_words = set(re.findall(r'\w+', doc))
        if name_words and name_words.issubset(doc_words) and len(doc_words) < len(name_words) + 4:
            self._issue("LOW", lineno, "REDUNDANT_DOCSTRING",
                        f"Docstring for '{func_name}' just restates the function name.",
                        "Add detail about the algorithm, edge cases, or return value contract.")

    def _check_bare_except(self, node):
        for handler in node.handlers:
            if handler.type is None:
                self._issue("MEDIUM", node.lineno, "BARE_EXCEPT",
                            "Bare 'except:' catches everything including KeyboardInterrupt.",
                            "Catch specific exceptions: 'except Exception:' at minimum.")

    def _check_redundant_try(self, node):
        # Try with a single-line body and a pass/return in except
        if len(node.body) == 1 and len(node.handlers) == 1:
            handler = node.handlers[0]
            if len(handler.body) == 1:
                stmt = handler.body[0]
                if isinstance(stmt, (ast.Pass, ast.Return, ast.Constant)):
                    # Single-line try with trivial except — possibly just silencing an error
                    pass  # Too many false positives to flag here

    # ------------------------------------------------------------------
    # Post-processing checks (run after full tree walk)
    # ------------------------------------------------------------------

    def run_post_checks(self):
        self._check_unused_imports()
        self._check_duplicate_functions()

    def _check_unused_imports(self):
        for name, lineno in self._imports.items():
            # Check top-level name and any dotted components
            base = name.split('.')[0]
            if base not in self._used_names and name not in self._used_names:
                self._issue("LOW", lineno, "UNUSED_IMPORT",
                            f"Import '{name}' does not appear to be used.",
                            f"Remove 'import {name}' if truly unused.")

    def _check_duplicate_functions(self):
        """Flag pairs of functions with very similar bodies (structural similarity)."""
        names = list(self._func_bodies.keys())
        flagged = set()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if a in flagged or b in flagged:
                    continue
                sim = _body_similarity(self._func_bodies[a], self._func_bodies[b])
                if sim > 0.85:
                    line_a = self._func_defs[a]
                    line_b = self._func_defs[b]
                    self._issue("MEDIUM", line_a, "DUPLICATE_FUNC",
                                f"'{a}' (line {line_a}) and '{b}' (line {line_b}) "
                                f"have {int(sim*100)}% structural similarity.",
                                f"Extract the shared logic into a single helper; parameterize the difference.")
                    flagged.add(b)


# ---------------------------------------------------------------------------
# Comment-based checks (token walk, not AST)
# ---------------------------------------------------------------------------

def check_comments(filename, source):
    issues = []
    lines = _src_lines(source)

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return issues

    for tok_type, tok_str, tok_start, _, _ in tokens:
        if tok_type != tokenize.COMMENT:
            continue
        lineno = tok_start[0]
        comment = tok_str.strip()

        # Separator decoration (pure dashes/equals)
        if re.match(r'^#+\s*[-=]{6,}\s*$', comment):
            issues.append(Issue("LOW", filename, lineno, "SEPARATOR_COMMENT",
                                f"Decorative separator comment: '{comment[:40]}'",
                                "Use blank lines for visual separation instead."))
            continue

        # Restate code
        for pattern in RESTATE_PATTERNS:
            if re.match(pattern, comment, re.IGNORECASE):
                next_line = _line_text(lines, lineno + 1) if lineno < len(lines) else ""
                issues.append(Issue("LOW", filename, lineno, "RESTATE_COMMENT",
                                    f"Comment may restate the next line: '{comment[:60]}'",
                                    "Remove or replace with a comment explaining WHY, not WHAT."))
                break

        # Generic comments
        for pattern in GENERIC_COMMENT_PATTERNS:
            if re.match(pattern, comment, re.IGNORECASE):
                issues.append(Issue("LOW", filename, lineno, "GENERIC_COMMENT",
                                    f"Generic comment: '{comment[:60]}'",
                                    "Be specific: explain the purpose or invariant, not the action."))
                break

        # Commented-out code (heuristic: line starts with # and looks like code)
        code_like = comment.lstrip('#').strip()
        if re.match(r'^(if|for|while|return|import|def |class |[a-zA-Z_]\w*\s*[=(\[])', code_like):
            if len(code_like) > 5:
                issues.append(Issue("LOW", filename, lineno, "COMMENTED_CODE",
                                    f"Possible commented-out code: '{comment[:60]}'",
                                    "Delete dead code; use version control to retrieve it if needed."))

        # Debug/logging statements in comments
        if re.search(r'\b(print|debug|logger\.|logging\.|breakpoint)\b', comment, re.IGNORECASE):
            issues.append(Issue("LOW", filename, lineno, "DEBUG_COMMENT",
                                f"Debug/logging reference in comment: '{comment[:60]}'",
                                "Remove debug references from comments in submitted code."))

    return issues


# ---------------------------------------------------------------------------
# Magic number check (token walk)
# ---------------------------------------------------------------------------

def check_magic_numbers(filename, source):
    issues = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return issues

    for i, (tok_type, tok_str, tok_start, _, _) in enumerate(tokens):
        if tok_type != tokenize.NUMBER:
            continue
        lineno = tok_start[0]
        try:
            val = float(tok_str)
        except ValueError:
            continue

        if val in MAGIC_NUMBER_WHITELIST:
            continue
        if val == int(val) and int(val) in MAGIC_NUMBER_WHITELIST:
            continue

        # Skip if previous non-whitespace token is '=' (it's a constant assignment)
        prev_meaningful = [t for t in tokens[:i]
                           if t[0] not in (tokenize.NL, tokenize.NEWLINE,
                                           tokenize.COMMENT, tokenize.INDENT,
                                           tokenize.DEDENT, tokenize.ENCODING)]
        if prev_meaningful and prev_meaningful[-1][1] in ('=', '==', '!='):
            continue

        # Skip if it's clearly a color/grid value (0–9 in ARC context)
        if 0 <= val <= 9:
            continue

        issues.append(Issue("LOW", filename, lineno, "MAGIC_NUMBER",
                            f"Magic number {tok_str} — consider a named constant.",
                            f"E.g.: MAX_PREDICTIONS = {tok_str}"))
    return issues


# ---------------------------------------------------------------------------
# Line-length and style checks
# ---------------------------------------------------------------------------

def check_style(filename, source):
    issues = []
    lines = _src_lines(source)
    for i, line in enumerate(lines, 1):
        # Long lines
        if len(line) > 110:
            issues.append(Issue("LOW", filename, i, "LONG_LINE",
                                f"Line is {len(line)} characters.",
                                "Keep lines ≤ 100 characters for readability."))
    return issues


# ---------------------------------------------------------------------------
# Nesting depth calculator
# ---------------------------------------------------------------------------

def _max_nesting_depth(func_node):
    """Return the maximum nesting depth of control-flow inside a function."""
    class DepthVisitor(ast.NodeVisitor):
        def __init__(self):
            self.max_depth = 0
            self.depth = 0

        def _enter(self, node):
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            self.generic_visit(node)
            self.depth -= 1

        visit_If = _enter
        visit_For = _enter
        visit_While = _enter
        visit_With = _enter
        visit_Try = _enter
        visit_ExceptHandler = _enter

    dv = DepthVisitor()
    for stmt in func_node.body:
        dv.visit(stmt)
    return dv.max_depth


# ---------------------------------------------------------------------------
# Name utilities
# ---------------------------------------------------------------------------

def _split_name(name):
    """Split snake_case or camelCase name into words."""
    # snake_case
    parts = name.split('_')
    parts = [p for p in parts if p]
    return parts


# ---------------------------------------------------------------------------
# Name collector helper
# ---------------------------------------------------------------------------

class NameCollector(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.names.add(node.value.id)
        self.names.add(node.attr)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Structural similarity for duplicate detection
# ---------------------------------------------------------------------------

def _normalize_body(lines):
    """Strip whitespace and variable names for structural comparison."""
    normalized = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Remove string literals
        line = re.sub(r'"[^"]*"|\'[^\']*\'', '""', line)
        # Normalize identifiers to placeholders (keep keywords/operators)
        line = re.sub(r'\b[a-z_][a-z0-9_]{3,}\b', 'X', line)
        normalized.append(line)
    return normalized


def _body_similarity(lines_a, lines_b):
    """Return 0.0–1.0 similarity score between two function bodies."""
    a = _normalize_body(lines_a)
    b = _normalize_body(lines_b)
    if not a or not b:
        return 0.0
    # Skip very short functions (likely legitimate small helpers)
    if len(a) < 4 or len(b) < 4:
        return 0.0
    # Jaccard similarity on line-sets
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# ARC-specific checks
# ---------------------------------------------------------------------------

def check_arc_specific(filename, source):
    """ARC-project-specific heuristics."""
    issues = []
    lines = _src_lines(source)

    # Check for hardcoded problem IDs
    id_pattern = re.compile(r'["\'][0-9a-f]{8}["\']')
    for i, line in enumerate(lines, 1):
        if id_pattern.search(line) and 'json' not in line.lower() and 'path' not in line.lower():
            issues.append(Issue("HIGH", filename, i, "HARDCODED_PROBLEM_ID",
                                f"Possible hardcoded ARC problem ID: '{line.strip()[:60]}'",
                                "Do not hardcode problem IDs. Use problem_name() only for debugging."))

    # Check for test output access
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if 'test_set()' in stripped and 'get_output_data' in stripped:
            issues.append(Issue("HIGH", filename, i, "TEST_OUTPUT_ACCESS",
                                "Agent accesses test output via test_set().get_output_data().",
                                "Test output is None on Gradescope. Remove this access."))

    # Check for network calls
    net_imports = ['urllib', 'requests', 'http.client', 'socket', 'httpx', 'aiohttp']
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = ''
                if isinstance(node, ast.Import):
                    mod = node.names[0].name
                else:
                    mod = node.module or ''
                for net in net_imports:
                    if mod.startswith(net):
                        issues.append(Issue("HIGH", filename, node.lineno, "NETWORK_IMPORT",
                                            f"Network library '{mod}' imported in agent.",
                                            "Gradescope has no network access. Remove this."))
    except SyntaxError:
        pass

    # Check for matplotlib import
    for i, line in enumerate(lines, 1):
        if re.match(r'\s*(import matplotlib|from matplotlib)', line):
            issues.append(Issue("HIGH", filename, i, "MATPLOTLIB_IMPORT",
                                "matplotlib imported. This will crash on Gradescope (headless).",
                                "Remove matplotlib from ArcAgent.py. Use it only in visualization scripts."))

    # Check prediction list cap
    for i, line in enumerate(lines, 1):
        if 'return predictions' in line and '[:3]' not in line and '[:' not in line:
            # Only warn if we're not truncating
            pass  # too many false positives to flag naively

    return issues


# ---------------------------------------------------------------------------
# Score comparison
# ---------------------------------------------------------------------------

def run_driver(driver_path):
    """Run ArcDriver.py and extract the score. Returns (correct, total) or None."""
    try:
        result = subprocess.run(
            [sys.executable, driver_path],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(driver_path)
        )
        output = result.stdout + result.stderr
        match = re.search(r'RESULT:\s*(\d+)/(\d+)', output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception as e:
        print(f"  [driver error] {e}")
    return None


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_file(filepath):
    with open(filepath, encoding='utf-8') as f:
        source = f.read()

    filename = os.path.basename(filepath)
    issues = []

    # Parse AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  [PARSE ERROR] {filename}: {e}")
        return issues

    # AST-based checks
    linter = ArcLinter(filename, source)
    linter.visit(tree)
    linter.run_post_checks()
    issues.extend(linter.issues)

    # Token-based checks
    issues.extend(check_comments(filename, source))
    issues.extend(check_magic_numbers(filename, source))
    issues.extend(check_style(filename, source))
    issues.extend(check_arc_specific(filename, source))

    # Sort by line number
    issues.sort(key=lambda x: (x.line, x.severity))
    return issues


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
SEVERITY_COUNTS = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}


def print_report(all_issues, files_analyzed, score_before, score_after):
    total = len(all_issues)
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for issue in all_issues:
        counts[issue.severity] += 1

    print()
    print("=" * 65)
    print("  CODE QUALITY REPORT — ARC-AGI Project")
    print("=" * 65)
    print(f"  Files analyzed:  {files_analyzed}")
    print(f"  Total issues:    {total}  "
          f"(HIGH: {counts['HIGH']}  MEDIUM: {counts['MEDIUM']}  LOW: {counts['LOW']})")
    if score_before is not None:
        b_c, b_t = score_before
        print(f"  Score before:    {b_c}/{b_t}")
    if score_after is not None:
        a_c, a_t = score_after
        print(f"  Score after:     {a_c}/{a_t}")
        if score_before:
            delta = a_c - b_c
            symbol = "+" if delta >= 0 else ""
            print(f"  Score delta:     {symbol}{delta}")
    print("=" * 65)

    # Group by file, then severity
    by_file = defaultdict(list)
    for issue in all_issues:
        by_file[issue.filename].append(issue)

    for filename, file_issues in sorted(by_file.items()):
        print(f"\n  FILE: {filename}")
        print(f"  {'─' * 60}")
        for severity in ("HIGH", "MEDIUM", "LOW"):
            for issue in sorted(file_issues, key=lambda x: x.line):
                if issue.severity == severity:
                    print(f"  {issue}")
        print()

    # Summary of recommended changes
    print("=" * 65)
    print("  RECOMMENDED ACTIONS")
    print("=" * 65)
    _print_recommendations(all_issues)

    print()
    print("  NOTE: No files were modified. Review before applying any fix.")
    print("        Run: python arc_lint.py --fix  to apply safe auto-fixes only.")
    print("        Always re-run ArcDriver.py and confirm score is preserved.")
    print()


def _print_recommendations(issues):
    seen_codes = set()
    n = 1
    priority = [
        ("HARDCODED_PROBLEM_ID", "Remove any hardcoded ARC problem IDs from the agent."),
        ("TEST_OUTPUT_ACCESS",   "Remove test output access — it returns None on Gradescope."),
        ("MATPLOTLIB_IMPORT",    "Remove matplotlib from any submitted file."),
        ("NETWORK_IMPORT",       "Remove network imports — Gradescope has no internet access."),
        ("LONG_FUNC",            "Split large functions into focused helpers (≤30 lines each)."),
        ("DUPLICATE_FUNC",       "Consolidate duplicate strategy functions into a parameterized helper."),
        ("DEEP_NESTING",         "Flatten deeply nested logic with early returns."),
        ("BARE_EXCEPT",          "Replace bare 'except:' with 'except Exception:'."),
        ("UNUSED_IMPORT",        "Remove unused imports."),
        ("UNUSED_PARAM",         "Remove or prefix unused parameters with '_'."),
        ("RESTATE_COMMENT",      "Remove comments that only restate what the code does."),
        ("GENERIC_COMMENT",      "Replace generic comments with specific explanations."),
        ("COMMENTED_CODE",       "Delete commented-out code; use git history instead."),
        ("MAGIC_NUMBER",         "Replace magic numbers with named constants at module top."),
        ("GENERIC_DOCSTRING",    "Improve docstrings to describe algorithm and edge cases."),
        ("VERBOSE_NAME",         "Shorten overly long function/variable names."),
        ("LONG_LINE",            "Break long lines for readability."),
        ("SEPARATOR_COMMENT",    "Remove decorative separator comments."),
    ]
    active_codes = {i.code for i in issues}
    for code, description in priority:
        if code in active_codes and code not in seen_codes:
            count = sum(1 for i in issues if i.code == code)
            print(f"  {n}. [{code}] {description}  ({count} occurrence{'s' if count > 1 else ''})")
            seen_codes.add(code)
            n += 1


# ---------------------------------------------------------------------------
# Safe auto-fix (--fix mode)
# ---------------------------------------------------------------------------

def safe_autofix(filepath, issues):
    """
    Apply only unambiguous, safe fixes:
    - Remove trailing whitespace
    - Remove pure separator comment lines (if flagged)
    Returns (new_source, fixes_applied list).
    """
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    fixes = []
    separator_lines = {i.line for i in issues
                       if i.code == "SEPARATOR_COMMENT" and i.filename == os.path.basename(filepath)}

    new_lines = []
    for i, line in enumerate(lines, 1):
        if i in separator_lines:
            fixes.append(f"  Line {i}: removed decorative separator comment")
            continue
        # trailing whitespace
        stripped = line.rstrip() + '\n'
        if stripped != line:
            fixes.append(f"  Line {i}: removed trailing whitespace")
            line = stripped
        new_lines.append(line)

    return ''.join(new_lines), fixes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    fix_mode = "--fix" in sys.argv
    run_score = "--score" in sys.argv or "--fix" in sys.argv

    # Determine files to analyze
    targets = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not targets:
        targets = [
            os.path.join(root, "ArcAgent.py"),
            os.path.join(root, "ArcDriver.py"),
            os.path.join(root, "ArcProblem.py"),
        ]
    else:
        targets = [os.path.join(root, t) if not os.path.isabs(t) else t for t in targets]

    targets = [t for t in targets if os.path.isfile(t)]
    if not targets:
        print("No target files found.")
        sys.exit(1)

    driver_path = os.path.join(root, "ArcDriver.py")

    # Score before (only if running driver makes sense)
    score_before = None
    if run_score and os.path.isfile(driver_path):
        print("  Running ArcDriver.py to get baseline score...")
        score_before = run_driver(driver_path)
        if score_before:
            print(f"  Baseline: {score_before[0]}/{score_before[1]}")

    # Analyze
    all_issues = []
    for filepath in targets:
        issues = analyze_file(filepath)
        all_issues.extend(issues)

    score_after = None

    # Apply safe fixes if requested
    if fix_mode:
        print("\n  Applying safe auto-fixes...")
        any_fixed = False
        for filepath in targets:
            file_issues = [i for i in all_issues if i.filename == os.path.basename(filepath)]
            new_source, fixes = safe_autofix(filepath, file_issues)
            if fixes:
                any_fixed = True
                print(f"\n  {os.path.basename(filepath)}:")
                for fix in fixes:
                    print(fix)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_source)
        if any_fixed and os.path.isfile(driver_path):
            print("\n  Re-running ArcDriver.py after fixes...")
            score_after = run_driver(driver_path)
            if score_after:
                print(f"  Score after fixes: {score_after[0]}/{score_after[1]}")

    # Print report
    print_report(all_issues, len(targets), score_before, score_after)


if __name__ == "__main__":
    main()
