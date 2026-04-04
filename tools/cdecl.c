/* cdecl.c — hoist C declarations to function top.
   Minimal C parser: understands functions, blocks, declarations.
   Everything else is opaque text passed through.

   Usage: cdecl [-f func] < input.c > output.c
          cdecl -l < input.c            (list functions)
*/
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAXSRC = 4 * 1024 * 1024, MAXDECL = 1024, MAXNAME = 64 };

static char src[MAXSRC];
static int srcn;

static int is_type_kw(const char *w) {
  static const char *kw[] = {"int",    "long",     "char",    "short",
                             "float",  "double",   "void",    "unsigned",
                             "signed", "size_t",   "int8_t",  "int16_t",
                             "int32_t","int64_t",  "uint8_t", "uint16_t",
                             "uint32_t","uint64_t","Real",    "enum",
                             "struct", "const",    "static",  "volatile",
                             NULL};
  for (int i = 0; kw[i]; i++)
    if (strcmp(w, kw[i]) == 0) return 1;
  return 0;
}

static int is_ident_char(int c) { return isalnum(c) || c == '_'; }

/* skip whitespace, return count */
static int skipws(int p) {
  while (p < srcn && isspace((unsigned char)src[p])) p++;
  return p;
}

/* read identifier at p into buf, return end position */
static int read_ident(int p, char *buf, int bufsz) {
  int i = 0;
  while (p < srcn && is_ident_char(src[p]) && i < bufsz - 1)
    buf[i++] = src[p++];
  buf[i] = 0;
  return p;
}

/* skip balanced parens/brackets/braces from p (p points to opener) */
static int skip_balanced(int p) {
  char open = src[p], close;
  int depth = 1;
  if (open == '(') close = ')';
  else if (open == '[') close = ']';
  else if (open == '{') close = '}';
  else return p + 1;
  p++;
  while (p < srcn && depth > 0) {
    if (src[p] == '"' || src[p] == '\'') {
      char q = src[p++];
      while (p < srcn && src[p] != q) { if (src[p] == '\\') p++; p++; }
      if (p < srcn) p++;
    } else {
      if (src[p] == open) depth++;
      else if (src[p] == close) depth--;
      p++;
    }
  }
  return p;
}

/* find matching } for { at pos */
static int find_close_brace(int p) { return skip_balanced(p); }

struct Decl {
  int start, end;  /* byte range in src */
  int has_init;    /* has = initializer */
  int has_brace;   /* initializer contains { */
  char type[128];  /* base type string */
  char name[MAXNAME]; /* variable name (without * or []) */
  char full_name[128]; /* full declarator e.g. "*p" or "buf[100]" */
};

struct Func {
  char name[MAXNAME];
  int name_pos;
  int body_start; /* position of { */
  int body_end;   /* position after } */
};

static struct Func funcs[256];
static int nfuncs;

/* try to parse a declaration at position p inside a block.
   returns end position if found, or 0 if not a declaration. */
static int try_parse_decl(int p, struct Decl *d) {
  int p0 = p;
  char word[128];
  int type_end;

  /* skip qualifiers and type keywords */
  p = skipws(p);
  type_end = p;
  while (1) {
    int wp = p;
    p = read_ident(p, word, sizeof word);
    if (word[0] == 0) break;
    if (!is_type_kw(word)) { p = wp; break; }
    p = skipws(p);
    /* handle 'struct Name' or 'enum Name' */
    if (strcmp(word, "struct") == 0 || strcmp(word, "enum") == 0) {
      p = read_ident(p, word, sizeof word);
      p = skipws(p);
      /* anonymous struct definition { ... } — not a declaration we hoist */
      if (p < srcn && src[p] == '{') return 0;
    }
    type_end = p;
  }
  if (type_end == p0 + (skipws(p0) - p0)) return 0; /* no type found */

  /* extract type string */
  { int ts = skipws(p0), te = type_end;
    while (te > ts && isspace((unsigned char)src[te-1])) te--;
    int len = te - ts;
    if (len <= 0 || len >= (int)sizeof(d->type)) return 0;
    memcpy(d->type, src + ts, len);
    d->type[len] = 0;
  }

  /* now expect declarator: optional *, name, optional [size] */
  p = skipws(p);
  int fi = 0;
  while (p < srcn && src[p] == '*') { d->full_name[fi++] = '*'; p++; p = skipws(p); }
  p = read_ident(p, word, sizeof word);
  if (word[0] == 0) return 0; /* no identifier — not a declaration */
  strncpy(d->name, word, MAXNAME - 1);
  for (int i = 0; word[i] && fi < 126; i++) d->full_name[fi++] = word[i];

  p = skipws(p);
  /* array suffix */
  if (p < srcn && src[p] == '[') {
    int bracket_start = p;
    p = skip_balanced(p);
    int blen = p - bracket_start;
    if (fi + blen < 126) { memcpy(d->full_name + fi, src + bracket_start, blen); fi += blen; }
    p = skipws(p);
  }
  d->full_name[fi] = 0;

  /* check for = initializer */
  d->has_init = 0;
  d->has_brace = 0;
  if (p < srcn && src[p] == '=') {
    d->has_init = 1;
    p++; p = skipws(p);
    if (p < srcn && src[p] == '{') d->has_brace = 1;
  }

  /* skip to ; or , */
  while (p < srcn && src[p] != ';' && src[p] != ',') {
    if (src[p] == '(' || src[p] == '[' || src[p] == '{') p = skip_balanced(p);
    else p++;
  }

  /* for now, handle single-declarator lines only (stop at ;) */
  /* skip multi-declarator (,) — just consume to ; */
  while (p < srcn && src[p] != ';') {
    if (src[p] == '(' || src[p] == '[' || src[p] == '{') p = skip_balanced(p);
    else p++;
  }
  if (p < srcn) p++; /* skip ; */

  /* skip trailing newline */
  if (p < srcn && src[p] == '\n') p++;

  d->start = p0;
  d->end = p;
  return p;
}

/* find all functions in src */
static void find_functions(void) {
  int p = 0;
  nfuncs = 0;
  while (p < srcn) {
    /* look for pattern: type name ( ... ) { */
    if (src[p] == '{') {
      /* check if this is a function body */
      /* scan backwards for ) */
      int q = p - 1;
      while (q >= 0 && isspace((unsigned char)src[q])) q--;
      if (q >= 0 && src[q] == ')') {
        /* find matching ( */
        int depth = 1, r = q - 1;
        while (r >= 0 && depth > 0) {
          if (src[r] == ')') depth++;
          else if (src[r] == '(') depth--;
          r--;
        }
        r++; /* r is at ( */
        /* scan back for function name */
        int ne = r - 1;
        while (ne >= 0 && isspace((unsigned char)src[ne])) ne--;
        int ns = ne;
        while (ns > 0 && is_ident_char(src[ns-1])) ns--;
        if (ns <= ne && ne >= 0) {
          int len = ne - ns + 1;
          if (len > 0 && len < MAXNAME && nfuncs < 256) {
            memcpy(funcs[nfuncs].name, src + ns, len);
            funcs[nfuncs].name[len] = 0;
            funcs[nfuncs].name_pos = ns;
            funcs[nfuncs].body_start = p;
            funcs[nfuncs].body_end = find_close_brace(p);
            nfuncs++;
          }
        }
      }
      p = funcs[nfuncs-1].body_end;
    } else {
      p++;
    }
  }
}

static void emit(const char *s, int len) { fwrite(s, 1, len, stdout); }

static void process_function(struct Func *f) {
  struct Decl decls[MAXDECL];
  int ndecl = 0;
  int body = f->body_start + 1; /* after { */
  int body_end = f->body_end - 1; /* before } */

  /* collect all declarations recursively */
  int p = body;
  while (p < body_end && ndecl < MAXDECL) {
    p = skipws(p);
    if (p >= body_end) break;
    /* try to parse a declaration */
    struct Decl d;
    int end = try_parse_decl(p, &d);
    if (end > 0 && !d.has_brace) {
      decls[ndecl++] = d;
      p = end;
    } else if (src[p] == '{') {
      /* enter block, continue looking for declarations */
      p++;
    } else if (src[p] == '}') {
      p++;
    } else if (strncmp(src + p, "for", 3) == 0 && !is_ident_char(src[p+3])) {
      /* skip for(...) but look inside the body */
      p += 3; p = skipws(p);
      if (src[p] == '(') p = skip_balanced(p);
      /* the body will be scanned by the outer loop */
    } else if (strncmp(src + p, "if", 2) == 0 && !is_ident_char(src[p+2])) {
      p += 2; p = skipws(p);
      if (src[p] == '(') p = skip_balanced(p);
    } else if (strncmp(src + p, "while", 5) == 0 && !is_ident_char(src[p+5])) {
      p += 5; p = skipws(p);
      if (src[p] == '(') p = skip_balanced(p);
    } else {
      /* skip to end of statement */
      while (p < body_end && src[p] != ';' && src[p] != '{' && src[p] != '}') {
        if (src[p] == '(' || src[p] == '[') p = skip_balanced(p);
        else p++;
      }
      if (p < body_end && src[p] == ';') p++;
    }
  }

  if (ndecl == 0) {
    emit(src + f->body_start, f->body_end - f->body_start);
    return;
  }

  /* deduplicate by name, keep first occurrence */
  int keep[MAXDECL];
  memset(keep, 1, sizeof keep);
  for (int i = 0; i < ndecl; i++) {
    if (!keep[i]) continue;
    for (int j = i + 1; j < ndecl; j++) {
      if (strcmp(decls[i].name, decls[j].name) == 0) {
        fprintf(stderr, "  warn: %s duplicate\n", decls[j].name);
        keep[j] = 0;
      }
    }
  }

  /* sort by type then name */
  int order[MAXDECL];
  for (int i = 0; i < ndecl; i++) order[i] = i;
  for (int i = 0; i < ndecl - 1; i++)
    for (int j = i + 1; j < ndecl; j++) {
      int ci = order[i], cj = order[j];
      int cmp = strcmp(decls[ci].type, decls[cj].type);
      if (cmp == 0) cmp = strcmp(decls[ci].full_name, decls[cj].full_name);
      if (cmp > 0) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    }

  /* emit { */
  emit("{\n", 2);

  /* emit grouped declarations */
  char prev_type[128] = "";
  int line_start = 1;
  for (int oi = 0; oi < ndecl; oi++) {
    int i = order[oi];
    if (!keep[i]) continue;
    if (strcmp(decls[i].type, prev_type) != 0) {
      if (!line_start) emit(";\n", 2);
      fprintf(stdout, "  %s %s", decls[i].type, decls[i].full_name);
      strcpy(prev_type, decls[i].type);
      line_start = 0;
    } else {
      fprintf(stdout, ", %s", decls[i].full_name);
    }
  }
  if (!line_start) emit(";\n", 2);

  /* emit body, replacing declarations with assignments */
  p = f->body_start + 1;
  while (p < f->body_end - 1) {
    /* check if current position matches a collected declaration */
    int found = -1;
    for (int i = 0; i < ndecl; i++) {
      if (decls[i].start == p) { found = i; break; }
    }
    if (found >= 0) {
      struct Decl *d = &decls[found];
      if (d->has_init) {
        /* emit assignment: name = value; */
        /* find = in the original text */
        int eq = d->start;
        while (eq < d->end && src[eq] != '=') eq++;
        if (eq < d->end) {
          /* emit indentation from original */
          int ls = d->start;
          while (ls < eq && isspace((unsigned char)src[ls])) { fputc(src[ls], stdout); ls++; }
          /* emit name = rest */
          fprintf(stdout, "%s ", d->name);
          emit(src + eq, d->end - eq);
        }
      }
      /* skip the declaration in source */
      p = d->end;
    } else {
      fputc(src[p], stdout);
      p++;
    }
  }

  /* emit } */
  emit("}\n", 2);
}

int main(int argc, char **argv) {
  char *only_func = NULL;
  int list_mode = 0;
  int i;

  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-f") == 0 && i + 1 < argc)
      only_func = argv[++i];
    else if (strcmp(argv[i], "-l") == 0)
      list_mode = 1;
  }

  srcn = fread(src, 1, MAXSRC - 1, stdin);
  src[srcn] = 0;

  find_functions();

  if (list_mode) {
    for (i = 0; i < nfuncs; i++)
      fprintf(stdout, "%4d  %s\n",
              funcs[i].body_end - funcs[i].body_start, funcs[i].name);
    return 0;
  }

  if (only_func) {
    /* emit everything, replacing only the target function */
    int last = 0;
    for (i = 0; i < nfuncs; i++) {
      if (strcmp(funcs[i].name, only_func) != 0) continue;
      emit(src + last, funcs[i].body_start - last);
      process_function(&funcs[i]);
      last = funcs[i].body_end;
    }
    emit(src + last, srcn - last);
  } else {
    /* process all functions */
    int last = 0;
    for (i = 0; i < nfuncs; i++) {
      emit(src + last, funcs[i].body_start - last);
      process_function(&funcs[i]);
      last = funcs[i].body_end;
    }
    emit(src + last, srcn - last);
  }
  return 0;
}
