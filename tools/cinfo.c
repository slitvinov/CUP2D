/* cinfo.c — print declaration info for each function.
   Shows what cdecl would do: which declarations would be hoisted,
   which are already at top, which are skipped.

   Usage: cinfo < input.c
          cinfo -f func < input.c
*/
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAXSRC = 4 * 1024 * 1024, MAXDECL = 1024, MAXNAME = 64 };

static char src[MAXSRC];
static int srcn;

static int is_type_kw(const char *w) {
  static const char *kw[] = {"int",     "long",     "char",     "short",
                             "float",   "double",   "void",     "unsigned",
                             "signed",  "size_t",   "int8_t",   "int16_t",
                             "int32_t", "int64_t",  "uint8_t",  "uint16_t",
                             "uint32_t","uint64_t", "Real",     "enum",
                             "struct",  "const",    "static",   "volatile",
                             NULL};
  int i;
  for (i = 0; kw[i]; i++)
    if (strcmp(w, kw[i]) == 0) return 1;
  return 0;
}

static int is_ident_char(int c) { return isalnum(c) || c == '_'; }

static int skipws(int p) {
  while (p < srcn && isspace((unsigned char)src[p])) p++;
  return p;
}

static int read_ident(int p, char *buf, int bufsz) {
  int i = 0;
  while (p < srcn && is_ident_char(src[p]) && i < bufsz - 1)
    buf[i++] = src[p++];
  buf[i] = 0;
  return p;
}

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

static int line_of(int pos) {
  int n = 1, i;
  for (i = 0; i < pos && i < srcn; i++)
    if (src[i] == '\n') n++;
  return n;
}

struct Decl {
  int start, end;
  int line;
  int depth;
  int has_init;
  int has_brace;
  char type[128];
  char name[MAXNAME];
  char full_name[128];
  char init_text[256];
};

struct Func {
  char name[MAXNAME];
  int body_start, body_end;
};

static struct Func funcs[256];
static int nfuncs;

static int try_parse_decl(int p, struct Decl *d, int depth) {
  int p0 = p;
  char word[128];
  int type_end;

  p = skipws(p);
  type_end = p;
  while (1) {
    int wp = p;
    p = read_ident(p, word, sizeof word);
    if (word[0] == 0) break;
    if (!is_type_kw(word)) { p = wp; break; }
    p = skipws(p);
    if (strcmp(word, "struct") == 0 || strcmp(word, "enum") == 0) {
      p = read_ident(p, word, sizeof word);
      p = skipws(p);
      if (p < srcn && src[p] == '{') return 0;
    }
    type_end = p;
  }
  if (type_end <= skipws(p0)) return 0;

  { int ts = skipws(p0), te = type_end;
    while (te > ts && isspace((unsigned char)src[te-1])) te--;
    int len = te - ts;
    if (len <= 0 || len >= (int)sizeof(d->type)) return 0;
    memcpy(d->type, src + ts, len);
    d->type[len] = 0;
  }

  p = skipws(p);
  int fi = 0;
  while (p < srcn && src[p] == '*') { d->full_name[fi++] = '*'; p++; p = skipws(p); }
  p = read_ident(p, word, sizeof word);
  if (word[0] == 0) return 0;
  strncpy(d->name, word, MAXNAME - 1);
  { int i; for (i = 0; word[i] && fi < 126; i++) d->full_name[fi++] = word[i]; }

  p = skipws(p);
  if (p < srcn && src[p] == '[') {
    int bs = p;
    p = skip_balanced(p);
    int blen = p - bs;
    if (fi + blen < 126) { memcpy(d->full_name + fi, src + bs, blen); fi += blen; }
    p = skipws(p);
  }
  d->full_name[fi] = 0;

  d->has_init = 0;
  d->has_brace = 0;
  d->init_text[0] = 0;
  if (p < srcn && src[p] == '=') {
    d->has_init = 1;
    int eq = p + 1;
    p++; p = skipws(p);
    if (p < srcn && src[p] == '{') d->has_brace = 1;
    int vs = p;
    while (p < srcn && src[p] != ';' && src[p] != ',') {
      if (src[p] == '(' || src[p] == '[' || src[p] == '{') p = skip_balanced(p);
      else p++;
    }
    int vlen = p - vs;
    if (vlen > 0 && vlen < 255) { memcpy(d->init_text, src + vs, vlen); d->init_text[vlen] = 0; }
  }

  while (p < srcn && src[p] != ';') {
    if (src[p] == '(' || src[p] == '[' || src[p] == '{') p = skip_balanced(p);
    else p++;
  }
  if (p < srcn) p++;
  if (p < srcn && src[p] == '\n') p++;

  d->start = p0;
  d->end = p;
  d->line = line_of(p0);
  d->depth = depth;
  return p;
}

static void collect_decls(int start, int end, int depth,
                          struct Decl *decls, int *ndecl) {
  int p = start;
  while (p < end && *ndecl < MAXDECL) {
    p = skipws(p);
    if (p >= end) break;
    struct Decl d;
    int r = try_parse_decl(p, &d, depth);
    if (r > 0 && !d.has_brace) {
      decls[(*ndecl)++] = d;
      p = r;
    } else if (src[p] == '{') {
      int close = skip_balanced(p);
      collect_decls(p + 1, close - 1, depth + 1, decls, ndecl);
      p = close;
    } else if (strncmp(src+p, "for", 3) == 0 && !is_ident_char(src[p+3])) {
      p += 3; p = skipws(p);
      if (p < end && src[p] == '(') p = skip_balanced(p);
    } else if (strncmp(src+p, "if", 2) == 0 && !is_ident_char(src[p+2])) {
      p += 2; p = skipws(p);
      if (p < end && src[p] == '(') p = skip_balanced(p);
    } else if (strncmp(src+p, "while", 5) == 0 && !is_ident_char(src[p+5])) {
      p += 5; p = skipws(p);
      if (p < end && src[p] == '(') p = skip_balanced(p);
    } else if (src[p] == '}') {
      p++;
    } else {
      while (p < end && src[p] != ';' && src[p] != '{' && src[p] != '}') {
        if (src[p] == '(' || src[p] == '[') p = skip_balanced(p);
        else p++;
      }
      if (p < end && src[p] == ';') p++;
    }
  }
}

static void find_functions(void) {
  int p = 0;
  nfuncs = 0;
  while (p < srcn) {
    if (src[p] == '{') {
      int q = p - 1;
      while (q >= 0 && isspace((unsigned char)src[q])) q--;
      if (q >= 0 && src[q] == ')') {
        int depth = 1, r = q - 1;
        while (r >= 0 && depth > 0) {
          if (src[r] == ')') depth++;
          else if (src[r] == '(') depth--;
          r--;
        }
        r++;
        int ne = r - 1;
        while (ne >= 0 && isspace((unsigned char)src[ne])) ne--;
        int ns = ne;
        while (ns > 0 && is_ident_char(src[ns-1])) ns--;
        if (ns <= ne && ne >= 0 && nfuncs < 256) {
          int len = ne - ns + 1;
          memcpy(funcs[nfuncs].name, src + ns, len);
          funcs[nfuncs].name[len] = 0;
          funcs[nfuncs].body_start = p;
          funcs[nfuncs].body_end = skip_balanced(p);
          nfuncs++;
          p = funcs[nfuncs-1].body_end;
          continue;
        }
      }
    }
    p++;
  }
}

/* count occurrences of identifier `name` in src[start..end) */
static int count_uses(const char *name, int start, int end) {
  int n = 0, nlen = strlen(name), p = start;
  while (p < end - nlen) {
    if (memcmp(src + p, name, nlen) == 0 &&
        (p == 0 || !is_ident_char(src[p - 1])) &&
        !is_ident_char(src[p + nlen])) {
      n++;
      p += nlen;
    } else {
      p++;
    }
  }
  return n;
}

/* find first use of identifier after `start` */
static int first_use_line(const char *name, int start, int end) {
  int nlen = strlen(name), p = start;
  while (p < end - nlen) {
    if (memcmp(src + p, name, nlen) == 0 &&
        (p == 0 || !is_ident_char(src[p - 1])) &&
        !is_ident_char(src[p + nlen]))
      return line_of(p);
    p++;
  }
  return -1;
}

static void report_function(struct Func *f) {
  struct Decl decls[MAXDECL];
  int ndecl = 0, i, j;
  int fline = line_of(f->body_start);
  int lines = line_of(f->body_end) - fline + 1;
  int n_top = 0, n_hoist = 0, n_dup = 0, n_warn = 0;

  collect_decls(f->body_start + 1, f->body_end - 1, 0, decls, &ndecl);

  if (ndecl == 0) return;

  /* check duplicates and shadows */
  int dup[MAXDECL], shadow[MAXDECL];
  memset(dup, 0, sizeof dup);
  memset(shadow, 0, sizeof shadow);
  for (i = 0; i < ndecl; i++) {
    for (j = i + 1; j < ndecl; j++) {
      if (strcmp(decls[i].name, decls[j].name) == 0) {
        dup[j] = i + 1;
        if (strcmp(decls[i].type, decls[j].type) != 0)
          shadow[j] = 1;
      }
    }
  }

  printf("%s (line %d, %d lines)\n", f->name, fline, lines);

  for (i = 0; i < ndecl; i++) {
    struct Decl *d = &decls[i];
    char flags[256] = "";
    int fl = 0;

    /* status */
    if (dup[i]) {
      fl += snprintf(flags + fl, sizeof(flags) - fl, "DUP");
      n_dup++;
    } else if (d->depth == 0 && !d->has_init) {
      fl += snprintf(flags + fl, sizeof(flags) - fl, "ok");
      n_top++;
    } else if (d->depth == 0 && d->has_init) {
      fl += snprintf(flags + fl, sizeof(flags) - fl, "SPLIT");
      n_hoist++;
    } else {
      fl += snprintf(flags + fl, sizeof(flags) - fl, "HOIST(d%d)", d->depth);
      n_hoist++;
    }

    /* unusual situations */
    if (shadow[i])
      fl += snprintf(flags + fl, sizeof(flags) - fl, " SHADOW(%s->%s)",
                     decls[dup[i]-1].type, d->type);

    if (d->has_brace)
      fl += snprintf(flags + fl, sizeof(flags) - fl, " BRACE_INIT");


    if (d->has_init && strstr(d->init_text, "->"))
      fl += snprintf(flags + fl, sizeof(flags) - fl, " MEMBER_INIT");

    /* check if unused (count uses in function body excluding declaration) */
    if (!dup[i]) {
      int uses = count_uses(d->name, f->body_start, f->body_end);
      if (uses <= 1)
        fl += snprintf(flags + fl, sizeof(flags) - fl, " UNUSED");
    }

    /* check late first use */
    if (!dup[i] && d->depth == 0) {
      int first = first_use_line(d->name, d->end, f->body_end);
      if (first > 0 && first - d->line > 30)
        fl += snprintf(flags + fl, sizeof(flags) - fl, " LATE_USE(+%d)",
                       first - d->line);
    }

    /* check for-loop variable that should stay */
    /* (detected by the parser as depth > 0 in for context — not easily
       distinguishable here, so skip) */

    printf("  %-12s %-18s %-15s line %-4d",
           flags, d->type, d->full_name, d->line);
    if (d->has_init) {
      char init_short[40];
      int ilen = strlen(d->init_text);
      if (ilen > 35) {
        memcpy(init_short, d->init_text, 32);
        strcpy(init_short + 32, "...");
      } else {
        strcpy(init_short, d->init_text);
      }
      printf(" = %s", init_short);
    }
    printf("\n");
    if (flags[0] != 'o') n_warn++;
  }

  printf("  --- %d ok, %d to hoist, %d dup, %d warnings\n\n",
         n_top, n_hoist, n_dup, n_warn);
}

int main(int argc, char **argv) {
  char *only_func = NULL;
  int i;

  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-f") == 0 && i + 1 < argc)
      only_func = argv[++i];
  }

  srcn = fread(src, 1, MAXSRC - 1, stdin);
  src[srcn] = 0;
  find_functions();

  for (i = 0; i < nfuncs; i++) {
    if (only_func && strcmp(funcs[i].name, only_func) != 0) continue;
    report_function(&funcs[i]);
  }
  return 0;
}
