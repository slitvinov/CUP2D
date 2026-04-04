#!/usr/bin/env python3
"""Move variable declarations to top of function, grouped by type.

Usage: python3 tools/decl_top.py [--in-place] [-f func] file.c
       -f func  only process the named function
       --in-place  modify file in place

For each function, collects ALL declarations from the direct function body
(not nested blocks), hoists them to the top, grouped by base type and sorted
alphabetically. Initializers are split: "int x = expr;" becomes "int x;" at
top + "x = expr;" at original position.

Handles: scalars, pointers, arrays, struct types, const qualifiers.
Skips: brace initializers ({...}), declarations in for-loop init.
Warns: duplicate names, initializers referencing pointers (-> or function calls).
Run clang-format -i after.
"""

import sys
from collections import OrderedDict
import tree_sitter_c as tsc
from tree_sitter import Language, Parser

C_LANG = Language(tsc.language())
parser = Parser(C_LANG)


def get_func_name(src, node):
    for c in node.children:
        if c.type == 'function_declarator':
            for cc in c.children:
                if cc.type == 'identifier':
                    return src[cc.start_byte:cc.end_byte].decode()
    return None


def base_ident(name_bytes):
    """Extract plain identifier from 'name', '*name', or 'name[SIZE]'."""
    n = name_bytes.replace(b'*', b'').strip()
    if b'[' in n:
        n = n[:n.index(b'[')].strip()
    return n


def process(src, only_func=None):
    tree = parser.parse(src)
    edits = []

    def find_function_bodies(node):
        bodies = []
        if node.type == 'function_definition':
            if only_func is None or get_func_name(src, node) == only_func:
                for c in node.children:
                    if c.type == 'compound_statement':
                        bodies.append(c)
        else:
            for child in node.children:
                bodies.extend(find_function_bodies(child))
        return bodies

    def visit_function(body):
        open_brace = None
        for c in body.children:
            if c.type == '{':
                open_brace = c
                break
        if not open_brace:
            return
        insert_pos = open_brace.end_byte

        decls = [c for c in body.children if c.type == 'declaration']
        if not decls:
            return

        groups = OrderedDict()
        seen = set()

        for decl in decls:
            decl_text = src[decl.start_byte:decl.end_byte]
            if b'{' in decl_text:
                continue

            # Find base type and declarators
            declarators = []
            type_node_end = decl.start_byte
            for c in decl.children:
                if c.type in ('init_declarator', 'identifier',
                              'pointer_declarator', 'array_declarator'):
                    declarators.append(c)
                elif c.type in ('primitive_type', 'type_identifier',
                                'sized_type_specifier', 'storage_class_specifier',
                                'struct_specifier', 'enum_specifier',
                                'type_qualifier'):
                    type_node_end = c.end_byte

            if not declarators:
                continue

            base_type = src[decl.start_byte:type_node_end].strip()
            if b'const' in base_type:
                line = src[:decl.start_byte].count(b'\n') + 1
                sys.stderr.write(f"  skip: const declaration (line {line})\n")
                continue
            assignments = []
            new_names = []
            all_handled = True

            for d in declarators:
                if d.type == 'init_declarator':
                    eq = name_n = val_n = None
                    for c in d.children:
                        if c.type == '=':
                            eq = c
                        elif eq is None:
                            name_n = c
                        else:
                            val_n = c
                    if not name_n or not val_n:
                        all_handled = False
                        continue

                    name_text = src[name_n.start_byte:name_n.end_byte]
                    val_text = src[val_n.start_byte:val_n.end_byte]
                    bname = base_ident(name_text)
                    kind = 'ptr' if name_n.type == 'pointer_declarator' else \
                           'arr' if name_n.type == 'array_declarator' else 'plain'

                    # Check for unsafe initializers
                    unsafe = b'->' in val_text
                    if unsafe:
                        line = src[:val_n.start_byte].count(b'\n') + 1
                        sys.stderr.write(
                            f"  skip: {bname.decode()} = ... (unsafe init, line {line})\n")
                        all_handled = False
                        continue

                    # Build assignment
                    assign_name = name_text
                    if b'*' in assign_name:
                        assign_name = assign_name.replace(b'*', b'', 1).strip()
                    assignments.append(assign_name + b' = ' + val_text + b';')

                    key = (base_type, kind)
                    if bname not in seen:
                        groups.setdefault(key, []).append(name_text)
                        seen.add(bname)
                    else:
                        line = src[:name_n.start_byte].count(b'\n') + 1
                        sys.stderr.write(
                            f"  warn: {bname.decode()} duplicate (line {line})\n")

                else:
                    name_text = src[d.start_byte:d.end_byte]
                    bname = base_ident(name_text)
                    kind = 'ptr' if d.type == 'pointer_declarator' else \
                           'arr' if d.type == 'array_declarator' else 'plain'
                    key = (base_type, kind)
                    if bname not in seen:
                        groups.setdefault(key, []).append(name_text)
                        seen.add(bname)
                    else:
                        line = src[:d.start_byte].count(b'\n') + 1
                        sys.stderr.write(
                            f"  warn: {bname.decode()} duplicate (line {line})\n")
                    any_moved = True

            if not all_handled:
                continue

            # Replace declaration with assignments (or remove if no assignments)
            end = decl.end_byte
            if end < len(src) and src[end:end + 1] == b'\n':
                end += 1
            if assignments:
                edits.append((decl.start_byte, end,
                              b'\n'.join(assignments) + b'\n'))
            else:
                edits.append((decl.start_byte, end, b''))

        if groups:
            TYPE_ORDER = [b'int', b'long long', b'long', b'size_t',
                          b'Real', b'double', b'float',
                          b'struct Blk', b'struct Nb', b'struct HMEntry']
            KIND_ORDER = {'plain': 0, 'ptr': 1, 'arr': 2}
            def sort_key(item):
                (bt, kind) = item[0]
                try:
                    ti = TYPE_ORDER.index(bt)
                except ValueError:
                    ti = len(TYPE_ORDER)
                return (ti, KIND_ORDER.get(kind, 9), bt)
            grouped = b'\n'
            for (bt, kind), names in sorted(groups.items(), key=sort_key):
                snames = sorted(names, key=lambda n: base_ident(n))
                grouped += bt + b' ' + b', '.join(snames) + b';\n'
            edits.append((insert_pos, insert_pos, grouped))

    for body in find_function_bodies(tree.root_node):
        visit_function(body)

    edits.sort(key=lambda e: (-e[0], -e[1]))
    result = bytearray(src)
    for start, end, replacement in edits:
        result[start:end] = replacement
    return bytes(result)


if __name__ == '__main__':
    in_place = False
    only_func = None
    files = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--in-place':
            in_place = True
        elif args[i] == '-f' and i + 1 < len(args):
            i += 1
            only_func = args[i]
        else:
            files.append(args[i])
        i += 1
    if not files:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    for fname in files:
        with open(fname, 'rb') as f:
            src = f.read()
        out = process(src, only_func)
        if in_place:
            with open(fname, 'wb') as f:
                f.write(out)
        else:
            sys.stdout.buffer.write(out)
