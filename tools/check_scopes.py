#!/usr/bin/env python3
"""Find names that are used but never defined, per function scope.

Written after the same bug shipped three times and cost three simulation runs:
a variable's usages get renamed but one is missed, or its definition moves, and the
script dies at runtime with NameError halfway through an attempt.

A plain module-wide "collect every assigned name, subtract every loaded name" check
does NOT catch this. If any function anywhere still has a parameter called thumb_yaw,
that name looks defined, so a stale reference to it inside a different function passes
unnoticed. That is exactly how `NameError: name 'thumb_yaw' is not defined` reached a
live run after the parameter was renamed to `lateral`.

symtable resolves names the way Python actually does -- per scope, with enclosing
function scopes handled properly -- so a reference that would raise NameError at runtime
is visible here without executing anything.

Usage: python3 tools/check_scopes.py FILE [FILE...]
Exits non-zero if anything is unresolved.
"""
import builtins
import sys
import symtable

BUILTINS = set(dir(builtins))


def module_globals(table):
    """Names bound at module level: defs, classes, imports, assignments."""
    return {n for n in table.get_identifiers()
            if table.lookup(n).is_assigned() or table.lookup(n).is_imported()
            or table.lookup(n).is_namespace()}


def walk(table, globals_, path, problems):
    for child in table.get_children():
        name = f"{path}.{child.get_name()}" if path else child.get_name()
        if child.get_type() == "function":
            for ident in child.get_identifiers():
                sym = child.lookup(ident)
                if not sym.is_referenced():
                    continue
                # Local, a parameter, or bound in an enclosing function: fine.
                if sym.is_local() or sym.is_parameter() or sym.is_free():
                    continue
                # Everything else resolves at module level or to a builtin.
                if ident in globals_ or ident in BUILTINS:
                    continue
                problems.append((name, ident))
        walk(child, globals_, name, problems)


def check(path):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    table = symtable.symtable(src, path, "exec")
    problems = []
    walk(table, module_globals(table), "", problems)
    for scope, ident in problems:
        print(f"{path}: {scope}() uses '{ident}', which is not defined in that scope "
              f"or at module level")
    return not problems


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    ok = all([check(p) for p in sys.argv[1:]])
    print("scope check: OK" if ok else "scope check: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
