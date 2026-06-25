"""Z3 concolic solver using CDT type info from AkaUT REST API.

No regex. All structural data (types, parameters, struct fields) comes from
CDT via /api/node/conditions and /api/context endpoints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from z3 import Bool, BoolVal, Int, IntVal, Not, Solver, sat

from covxplore.api_client import AkaUTClient, ConditionInfo, ConditionsResult

_console = Console()

# Simple type → default init value (for struct field init)
_SIMPLE = {"int", "size_t", "short", "long", "char", "unsigned char", "float", "double"}
_DEFAULT: dict[str, str] = {
    "int": "0", "size_t": "0", "short": "0", "long": "0",
    "char": "0", "bool": "false", "float": "0.0f", "double": "0.0",
}

# ---------------------------------------------------------------------------
# Condition parsing (regex for operator extraction, CDT for types)
# ---------------------------------------------------------------------------

_RE_COMPARE = re.compile(
    r"(\w+(?:->\w+)*(?:\.\w+)*)\s*(<=|>=|!=|==|<|>)\s*(\d+)"
)
_RE_BOOL_VAR = re.compile(r"^!?(\w+(?:->\w+)*(?:\.\w+)*)$")


@dataclass(frozen=True)
class ParsedCondition:
    node_id: int | None
    raw: str
    var_name: str      # Z3-safe name
    member_name: str   # C++ field/param name
    op: str            # < <= > >= == != bool !bool
    rhs: int           # constant (unused for bool)
    display: str
    z3_type: str       # "Int" or "Bool"


def _sanitise(name: str) -> str:
    return name.replace("->", "_").replace(".", "_")


def _resolve_var_type(var_raw: str, vars_map: dict[str, str]) -> str:
    if var_raw in vars_map:
        return vars_map[var_raw]
    if var_raw.lstrip("*") in vars_map:
        return vars_map[var_raw.lstrip("*")]
    return ""


def _to_z3_type(cpp_type: str) -> str | None:
    lower = cpp_type.lower()
    if any(t in lower for t in ("int", "size_t", "short", "long", "char")):
        return "Int"
    if "bool" in lower:
        return "Bool"
    if "float" in lower or "double" in lower:
        return "Int"
    return None


def parse_conditions(results: list[ConditionInfo]) -> list[ParsedCondition]:
    parsed: list[ParsedCondition] = []
    for info in results:
        text = info.condition.strip()
        vars_map = info.variables or {}

        # Comparison pattern: "var op const"
        m = _RE_COMPARE.search(text)
        if m:
            var_raw, op, rhs_str = m.group(1), m.group(2), m.group(3)
            var_type = _resolve_var_type(var_raw, vars_map)
            z3_type = _to_z3_type(var_type)
            if z3_type is None:
                continue
            parsed.append(ParsedCondition(
                node_id=info.node_id, raw=text,
                var_name=_sanitise(var_raw),
                member_name=var_raw.split("->")[-1].split(".")[-1],
                op=op, rhs=int(rhs_str),
                display=f"{var_raw} {op} {int(rhs_str)}",
                z3_type=z3_type,
            ))
            continue

        # Pure boolean: "var" or "!var"
        m = _RE_BOOL_VAR.match(text)
        if m:
            var_raw = m.group(1)
            negated = text.startswith("!")
            var_type = _resolve_var_type(var_raw, vars_map)
            if "bool" not in var_type.lower():
                continue
            parsed.append(ParsedCondition(
                node_id=info.node_id, raw=text,
                var_name=_sanitise(var_raw),
                member_name=var_raw.split("->")[-1].split(".")[-1],
                op="!bool" if negated else "bool", rhs=0,
                display=f"{var_raw} == {'true' if not negated else 'false'}",
                z3_type="Bool",
            ))

    return parsed


# ---------------------------------------------------------------------------
# Z3 solving
# ---------------------------------------------------------------------------

def _to_z3(zvar: Any, op: str, rhs: int) -> Any:
    if op == "<":   return zvar < IntVal(rhs)
    if op == "<=":  return zvar <= IntVal(rhs)
    if op == ">":   return zvar > IntVal(rhs)
    if op == ">=":  return zvar >= IntVal(rhs)
    if op == "==":  return zvar == IntVal(rhs)
    if op == "!=":  return zvar != IntVal(rhs)
    return BoolVal(False)


def solve_pair(
    cond: ParsedCondition,
) -> tuple[int | None, int | None] | tuple[bool | None, bool | None]:
    if cond.op == "bool":
        return (True, False)
    if cond.op == "!bool":
        return (False, True)

    zvar = Bool(cond.var_name) if cond.z3_type == "Bool" else Int(cond.var_name)
    constraint = _to_z3(zvar, cond.op, cond.rhs)

    s = Solver()
    s.add(constraint)
    true_val: int | bool | None = None
    if s.check() == sat:
        val = s.model()[zvar]
        true_val = bool(val) if cond.z3_type == "Bool" else val.as_long()

    s.reset()
    s.add(Not(constraint))
    false_val: int | bool | None = None
    if s.check() == sat:
        val = s.model()[zvar]
        false_val = bool(val) if cond.z3_type == "Bool" else val.as_long()

    return true_val, false_val


# ---------------------------------------------------------------------------
# Test body generation (CDT-driven — no regex)
# ---------------------------------------------------------------------------

def _cpp_val(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _build_test_body(
    params: list[dict[str, str]],
    qualified_name: str,
    all_variables: dict[str, str],  # condition var→type for struct field init
    overrides: dict[str, str],  # param/field_name → Z3 value
    polarity: str,
    target_desc: str,
) -> str:
    """Generate test body. Handles struct params (init fields from CDT variables) and scalar params."""
    lines = [f"// Target: {target_desc} → {polarity} branch"]

    scalar_args: list[str] = []

    for p in params:
        pname = p["name"]
        ptype = p.get("type", "unknown")
        is_ptr = "*" in ptype

        if is_ptr and not _is_scalar_ptr(ptype):
            # Struct pointer — declare struct and init fields from CDT variables
            struct_type = ptype.replace("*", "").strip()
            # Try to resolve qualified struct type from variables
            for vk, vt in all_variables.items():
                if (vk == pname or vk == pname + " *") and ("::" in vt or " " in vt):
                    struct_type = vt.replace("*", "").strip()
                    break
            struct_var = f"AKA_AI_{pname}"
            lines.append(f"{struct_type} {struct_var};")
            # Init fields from CDT variables that start with "p->" or "p."
            field_prefix = pname + "->"
            for vk, vt in all_variables.items():
                if vk.startswith(field_prefix):
                    fname = vk[len(field_prefix):]
                    if fname and _is_simple(vt):
                        val = overrides.get(fname,
                            overrides.get(vk, _DEFAULT.get(_short_type(vt), "0")))
                        lines.append(f"{struct_var}.{fname} = {val};")
                    elif fname and "char" in vt.lower() and "*" in vt:
                        # Pointer field → dummy array
                        lines.insert(1, "unsigned char AKA_AI_data[] = { 'a' };")
                        lines.append(f"{struct_var}.{fname} = AKA_AI_data;")
            scalar_args.append(f"&{struct_var}")
        else:
            # Scalar param
            cpp_type = _to_cpp_type(ptype)
            value = overrides.get(pname, "0")
            var = f"AKA_AI_{pname}"
            lines.append(f"{cpp_type} {var} = {value};")
            scalar_args.append(var)

    call_args = ", ".join(scalar_args)
    lines.append(f"bool AKA_AI_result = {qualified_name}({call_args});")
    return "\n".join(lines)


def _is_simple(vtype: str) -> bool:
    return _short_type(vtype) in _SIMPLE or "bool" in vtype.lower()


def _short_type(vtype: str) -> str:
    lower = vtype.lower()
    if "int" in lower: return "int"
    if "bool" in lower: return "bool"
    if "char" in lower: return "char"
    if "float" in lower or "double" in lower: return "float"
    return "other"


def _to_cpp_type(vtype: str) -> str:
    lower = vtype.lower()
    if "bool" in lower: return "bool"
    if "int" in lower: return "int"
    if "float" in lower or "double" in lower: return "double"
    if "char" in lower: return "char"
    return vtype.strip().replace("*", "").strip()


_SCALAR_PTR = {"char*", "int*", "bool*", "float*", "double*", "void*",
               "char *", "int *", "bool *", "float *", "double *", "void *"}

def _is_scalar_ptr(ptype: str) -> bool:
    """Check if a pointer type is a scalar (int*, char*, etc.) vs struct pointer."""
    clean = ptype.replace("const ", "").strip().lower()
    return clean in _SCALAR_PTR


def _default_init(ptype: str) -> str:
    lower = ptype.lower()
    if "bool" in lower: return "false"
    if any(t in lower for t in ("int", "size_t", "short", "long", "char", "float", "double")):
        return "0"
    return "0"


def _cpp_type(ptype: str) -> str:
    """Simplify CDT type to C++ declaration type."""
    lower = ptype.lower()
    if "bool" in lower: return "bool"
    if "int" in lower: return "int"
    if "float" in lower or "double" in lower: return "double"
    if "char" in lower: return "char"
    return ptype.strip()


# ---------------------------------------------------------------------------
# Pre-seed orchestrator
# ---------------------------------------------------------------------------

def pre_seed_z3_tests(
    function_path: str,
) -> list[dict[str, Any]]:
    """Fetch conditions + params from AkaUT CDT, solve with Z3, generate test bodies."""
    try:
        with AkaUTClient() as client:
            result: ConditionsResult = client.get_node_conditions(function_path)
    except Exception as exc:
        _console.print(f"[dim]Z3 pre-seed: condition fetch failed ({exc})[/]")
        return []

    parsed = parse_conditions(result.conditions)
    if not parsed:
        _console.print("[dim]Z3 pre-seed: no solvable conditions[/]")
        return []

    params = result.parameters or []
    # Build merged variables map from all conditions for struct field init
    all_vars: dict[str, str] = {}
    for c in result.conditions:
        if c.variables:
            all_vars.update(c.variables)
    # Extract qualified name from path: ".../Ns::func(...)" → "Ns::func"
    last_seg = function_path.split("/")[-1]
    m = re.match(r"^(.+?)\(.+$", last_seg)
    func_short = m.group(1) if m else last_seg
    ns = function_path.split("/")[-2] if "/" in function_path else ""
    qualified = f"{ns}::{func_short}" if ns else func_short

    tests: list[dict[str, Any]] = []
    solved_count = 0

    for cond in parsed:
        true_val, false_val = solve_pair(cond)

        if true_val is not None:
            body = _build_test_body(params, qualified, all_vars,
                                    {cond.member_name: _cpp_val(true_val)},
                                    "TRUE", f"Z3: {cond.display}")
            tests.append({
                "test_name": f"z3_cond{cond.node_id}_true",
                "test_body": body, "node_id": cond.node_id,
                "polarity": "TRUE",
                "target_reason": f"Z3 solved {cond.display} → TRUE = {true_val}",
            })
            solved_count += 1

        if false_val is not None:
            body = _build_test_body(params, qualified, all_vars,
                                    {cond.member_name: _cpp_val(false_val)},
                                    "FALSE", f"Z3: {cond.display}")
            tests.append({
                "test_name": f"z3_cond{cond.node_id}_false",
                "test_body": body, "node_id": cond.node_id,
                "polarity": "FALSE",
                "target_reason": f"Z3 solved {cond.display} → FALSE = {false_val}",
            })
            solved_count += 1

    _console.print(
        f"[green]Z3 pre-seed: {solved_count} test bodies from "
        f"{len(parsed)} solvable condition(s)[/]"
    )
    return tests
