#!/usr/bin/env python3
"""Move variable declarations to top of function, grouped by type and kind.

Usage: python3 tools/decl_top.py --in-place file.c
       python3 tools/decl_top.py file.c > file_fixed.c

Groups: Real a, b; Real *p, *q; int arr[10], brr[10];
Sorted alphabetically within each group.
Only hoists from direct function body (not nested blocks).
Skips: struct/enum decls, initializers with -> . or () calls, for-loop vars.
Run clang-format after.
"""

import sys
from collections import OrderedDict
import tree_sitter_c as tsc
from tree_sitter import Language, Parser

C_LANG = Language(tsc.language())
parser = Parser(C_LANG)


def process(src):
    tree = parser.parse(src)
    edits = []

    def find_function_bodies(node):
        bodies = []
        if node.type == 'function_definition':
            for c in node.children:
                if c.type == 'compound_statement':
                    bodies.append(c)
        else:
            for child in node.children:
                bodies.extend(find_function_bodies(child))
        return bodies

    def collect_for_decls(node, result):
        for child in node.children:
            if child.type == 'for_statement':
                for c in child.children:
                    if c.type == 'declaration':
                        result.append((child, c))
                        break
                for c in child.children:
                    if c.type == 'compound_statement':
                        collect_for_decls(c, result)
            elif child.type in ('compound_statement', 'if_statement',
                                'while_statement', 'do_statement'):
                collect_for_decls(child, result)

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

        for_decls = []
        collect_for_decls(body, for_decls)

        if not decls and not for_decls:
            return

        groups = OrderedDict()
        seen = set()

        for decl in decls:
            decl_text = src[decl.start_byte:decl.end_byte]
            if b'struct ' in decl_text or b'enum ' in decl_text:
                continue
            if b'{' in decl_text or b'[]' in decl_text:
                continue

            declarators = []
            type_node_end = decl.start_byte
            for c in decl.children:
                if c.type in ('init_declarator', 'identifier',
                              'pointer_declarator', 'array_declarator'):
                    declarators.append(c)
                elif c.type in ('primitive_type', 'type_identifier',
                                'sized_type_specifier', 'storage_class_specifier'):
                    type_node_end = c.end_byte

            if not declarators:
                continue

            base_type = src[decl.start_byte:type_node_end].strip()
            any_moved = False

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
                        continue
                    val_text = src[val_n.start_byte:val_n.end_byte]
                    if b'->' in val_text:
                        continue
                    has_call = any(c.type == 'call_expression'
                                  for c in val_n.children) if val_n.child_count else False
                    if not has_call:
                        has_call = val_n.type == 'call_expression'
                    if has_call:
                        continue
                    name_text = src[name_n.start_byte:name_n.end_byte]
                    kind = 'ptr' if name_n.type == 'pointer_declarator' else \
                           'arr' if name_n.type == 'array_declarator' else 'plain'
                    key = (base_type, kind)
                    if name_text not in seen:
                        groups.setdefault(key, []).append(name_text)
                        seen.add(name_text)
                    any_moved = True
                else:
                    name_text = src[d.start_byte:d.end_byte]
                    kind = 'ptr' if d.type == 'pointer_declarator' else \
                           'arr' if d.type == 'array_declarator' else 'plain'
                    key = (base_type, kind)
                    if name_text not in seen:
                        groups.setdefault(key, []).append(name_text)
                        seen.add(name_text)
                    any_moved = True

            if any_moved:
                assignments = []
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
                        if name_n and val_n:
                            val_text = src[val_n.start_byte:val_n.end_byte]
                            if b'->' in val_text or b'(' in val_text:
                                continue
                            name_raw = src[name_n.start_byte:name_n.end_byte]
                            if b'*' in name_raw:
                                name_assign = name_raw.replace(b'*', b'', 1)
                            else:
                                name_assign = name_raw
                            assignments.append(name_assign + b' = ' + val_text + b';')

                end = decl.end_byte
                if end < len(src) and src[end:end + 1] == b'\n':
                    end += 1

                if assignments:
                    edits.append((decl.start_byte, end,
                                  b'\n'.join(assignments) + b'\n'))
                else:
                    edits.append((decl.start_byte, end, b''))

        for for_stmt, for_decl in for_decls:
            declarators = []
            type_node_end = for_decl.start_byte
            for c in for_decl.children:
                if c.type in ('init_declarator', 'identifier'):
                    declarators.append(c)
                elif c.type in ('primitive_type', 'type_identifier',
                                'sized_type_specifier'):
                    type_node_end = c.end_byte
            base_type = src[for_decl.start_byte:type_node_end].strip()
            assignments = []
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
                    if name_n and val_n:
                        name_text = src[name_n.start_byte:name_n.end_byte]
                        val_text = src[val_n.start_byte:val_n.end_byte]
                        key = (base_type, 'plain')
                        if name_text not in seen:
                            groups.setdefault(key, []).append(name_text)
                            seen.add(name_text)
                        assignments.append(name_text + b' = ' + val_text)
                elif d.type == 'identifier':
                    name_text = src[d.start_byte:d.end_byte]
                    key = (base_type, 'plain')
                    if name_text not in seen:
                        groups.setdefault(key, []).append(name_text)
                        seen.add(name_text)
            if assignments:
                repl = b', '.join(assignments)
                end = for_decl.end_byte
                if end < len(src) and src[end:end+1] != b';':
                    pass
                else:
                    end += 1
                edits.append((for_decl.start_byte, end, repl + b';'))

        if groups:
            grouped = b'\n'
            for (base_type, kind), names in groups.items():
                snames = sorted(names)
                grouped += base_type + b' ' + b', '.join(snames) + b';\n'
            edits.append((insert_pos, insert_pos, grouped))

    for body in find_function_bodies(tree.root_node):
        visit_function(body)

    edits.sort(key=lambda e: (-e[0], -e[1]))
    result = bytearray(src)
    for start, end, replacement in edits:
        result[start:end] = replacement
    return bytes(result)


if __name__ == '__main__':
    in_place = '--in-place' in sys.argv
    files = [a for a in sys.argv[1:] if a != '--in-place']
    if not files:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    for fname in files:
        with open(fname, 'rb') as f:
            src = f.read()
        out = process(src)
        if in_place:
            with open(fname, 'wb') as f:
                f.write(out)
        else:
            sys.stdout.buffer.write(out)
