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


def walk(table, globals_, path, problems, shadowable):
    for child in table.get_children():
        name = f"{path}.{child.get_name()}" if path else child.get_name()
        if child.get_type() == "function":
            for ident in child.get_identifiers():
                sym = child.lookup(ident)
                if not sym.is_referenced():
                    continue
                # A local variable with the same name as a module-level function, class
                # or import makes that name local for the WHOLE function, so any use of
                # the module-level one inside it raises UnboundLocalError. This reached a
                # live run: attempt() called the helper sim_now() and later assigned a
                # variable called sim_now.
                if (sym.is_local() and sym.is_assigned() and not sym.is_parameter()
                        and ident in shadowable):
                    problems.append((name, ident, "shadow"))
                    continue
                # Local, a parameter, or bound in an enclosing function: fine.
                if sym.is_local() or sym.is_parameter() or sym.is_free():
                    continue
                # Everything else resolves at module level or to a builtin.
                if ident in globals_ or ident in BUILTINS:
                    continue
                problems.append((name, ident, "undefined"))
        walk(child, globals_, name, problems, shadowable)


def check(path):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    table = symtable.symtable(src, path, "exec")
    problems = []
    shadowable = {n for n in table.get_identifiers()
                  if table.lookup(n).is_namespace() or table.lookup(n).is_imported()}
    walk(table, module_globals(table), "", problems, shadowable)
    for scope, ident, kind in problems:
        if kind == "shadow":
            print(f"{path}: {scope}() assigns a local variable '{ident}', which hides the "
                  f"module-level function/import of that name inside the whole function")
        else:
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
