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
#include <unistd.h>

enum { MAXSRC = 4 * 1024 * 1024, MAXDECL = 1024, MAXNAME = 64 };

static char src[MAXSRC];
static int srcn;

static int is_type_kw(const char *w) {
  static const char *kw[] = {
      "int",      "long",     "char",     "short",    "float",
      "double",   "void",     "unsigned", "signed",   "size_t",
      "int8_t",   "int16_t",  "int32_t",  "int64_t",  "uint8_t",
      "uint16_t", "uint32_t", "uint64_t", "Real",     "enum",
      "struct",   "const",    "static",   "volatile", "FILE",
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
  else if (open == '[')
    close = ']';
  else if (open == '{')
    close = '}';
  else
    return p + 1;
  p++;
  while (p < srcn && depth > 0) {
    if (src[p] == '"' || src[p] == '\'') {
      char q = src[p++];
      while (p < srcn && src[p] != q) {
        if (src[p] == '\\') p++;
        p++;
      }
      if (p < srcn) p++;
    } else {
      if (src[p] == open) depth++;
      else if (src[p] == close)
        depth--;
      p++;
    }
  }
  return p;
}

/* find matching } for { at pos */
static int find_close_brace(int p) { return skip_balanced(p); }

struct Decl {
  int start, end;      /* byte range in src */
  int has_init;        /* has = initializer */
  int has_brace;       /* initializer contains { */
  int is_for;          /* declaration inside for() init */
  int is_vla;          /* array size is not constant */
  int val_end;         /* end of this declarator's value (at , or ;) */
  char type[128];      /* base type string */
  char name[MAXNAME];  /* variable name (without * or []) */
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

/* parse type prefix at p, return position after type keywords.
   fills type_str. returns 0 if no type found. */
static int parse_type(int p, char *type_str, int type_sz) {
  int p0 = p;
  char word[128];
  int type_end;

  p = skipws(p);
  type_end = p;
  while (1) {
    int wp = p;
    p = read_ident(p, word, sizeof word);
    if (word[0] == 0) break;
    if (!is_type_kw(word)) {
      p = wp;
      break;
    }
    p = skipws(p);
    if (strcmp(word, "struct") == 0 || strcmp(word, "enum") == 0) {
      p = read_ident(p, word, sizeof word);
      p = skipws(p);
      if (p < srcn && src[p] == '{') return 0;
    }
    type_end = p;
  }
  if (type_end <= skipws(p0)) return 0;

  {
    int ts = skipws(p0), te = type_end;
    while (te > ts && isspace((unsigned char)src[te - 1])) te--;
    int len = te - ts;
    if (len <= 0 || len >= type_sz) return 0;
    memcpy(type_str, src + ts, len);
    type_str[len] = 0;
  }
  return type_end;
}

/* parse one declarator at p (after type): optional *, name, optional [].
   fills d->name, d->full_name, d->has_init, d->has_brace.
   returns position after this declarator (at , or ;), or 0 on failure. */
static int parse_declarator(int p, struct Decl *d) {
  char word[128];
  int fi = 0;

  p = skipws(p);
  while (p < srcn && src[p] == '*') {
    d->full_name[fi++] = '*';
    p++;
    p = skipws(p);
  }
  p = read_ident(p, word, sizeof word);
  if (word[0] == 0) return 0;
  strncpy(d->name, word, MAXNAME - 1);
  {
    int i;
    for (i = 0; word[i] && fi < 126; i++) d->full_name[fi++] = word[i];
  }

  d->is_vla = 0;
  p = skipws(p);
  while (p < srcn && src[p] == '[') {
    int bs = p;
    p = skip_balanced(p);
    int blen = p - bs;
    /* check if array size contains variables (lowercase = VLA) */
    { int k;
      for (k = bs + 1; k < p - 1; k++)
        if (islower((unsigned char)src[k])) d->is_vla = 1;
    }
    if (fi + blen < 126) {
      memcpy(d->full_name + fi, src + bs, blen);
      fi += blen;
    }
    p = skipws(p);
  }
  d->full_name[fi] = 0;

  d->has_init = 0;
  d->has_brace = 0;
  if (p < srcn && src[p] == '=') {
    d->has_init = 1;
    p++;
    p = skipws(p);
    if (p < srcn && src[p] == '{') d->has_brace = 1;
  }

  /* skip to , or ; */
  while (p < srcn && src[p] != ';' && src[p] != ',') {
    if (src[p] == '(' || src[p] == '[' || src[p] == '{') p = skip_balanced(p);
    else
      p++;
  }
  return p;
}

/* try to parse declaration(s) at p. may produce multiple Decl for
   "int a, b, c;". returns end position, or 0 if not a declaration.
   *ndecl_out receives number of declarations added to d[]. */
static int try_parse_decls(int p, struct Decl *d, int maxd, int *ndecl_out) {
  int p0 = p;
  char type_str[128];
  int nd = 0;

  int tp = parse_type(p, type_str, sizeof type_str);
  if (tp == 0) return 0;
  p = tp;

  /* parse comma-separated declarators */
  while (nd < maxd) {
    struct Decl dd;
    memset(&dd, 0, sizeof dd);
    strcpy(dd.type, type_str);
    int dp = parse_declarator(p, &dd);
    if (dp == 0) return 0;
    dd.start = p0;
    dd.val_end = dp; /* at , or ; for this declarator */
    d[nd++] = dd;
    p = dp;
    if (p < srcn && src[p] == ',') {
      p++;
      continue;
    }
    break;
  }

  if (p < srcn && src[p] == ';') p++;
  if (p < srcn && src[p] == '\n') p++;

  /* set end for all declarators to the full statement end */
  {
    int i;
    for (i = 0; i < nd; i++) d[i].end = p;
  }
  *ndecl_out = nd;
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
          else if (src[r] == '(')
            depth--;
          r--;
        }
        r++; /* r is at ( */
        /* scan back for function name */
        int ne = r - 1;
        while (ne >= 0 && isspace((unsigned char)src[ne])) ne--;
        int ns = ne;
        while (ns > 0 && is_ident_char(src[ns - 1])) ns--;
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
      if (nfuncs > 0 && funcs[nfuncs - 1].body_start == p)
        p = funcs[nfuncs - 1].body_end;
      else
        p = skip_balanced(p); /* skip non-function { } block */
    } else {
      p++;
    }
  }
}

static FILE *outfp;
static void emit(const char *s, int len) { fwrite(s, 1, len, outfp); }

static void process_function(struct Func *f) {
  struct Decl decls[MAXDECL];
  int ndecl = 0;
  int body = f->body_start + 1;   /* after { */
  int body_end = f->body_end - 1; /* before } */

  /* collect all declarations recursively */
  int p = body;
  while (p < body_end && ndecl < MAXDECL) {
    p = skipws(p);
    if (p >= body_end) break;
    /* try to parse a declaration */
    struct Decl dd[32];
    int ndd = 0;
    int end = try_parse_decls(p, dd, 32, &ndd);
    if (end > 0 && ndd > 0 && !dd[0].has_brace) {
      /* if any declarator is VLA, skip the whole line */
      int k, has_vla = 0;
      for (k = 0; k < ndd; k++)
        if (dd[k].is_vla) has_vla = 1;
      if (!has_vla)
        for (k = 0; k < ndd && ndecl < MAXDECL; k++) decls[ndecl++] = dd[k];
      p = end;
    } else if (src[p] == '{') {
      /* enter block, continue looking for declarations */
      p++;
    } else if (src[p] == '}') {
      p++;
    } else if (strncmp(src + p, "for", 3) == 0 && !is_ident_char(src[p + 3])) {
      p += 3;
      p = skipws(p);
      if (p < body_end && src[p] == '(') {
        int paren_end = skip_balanced(p);
        int fp = p + 1; /* after ( */
        fp = skipws(fp);
        /* try to parse declaration in for-init */
        char type_str[128];
        int tp = parse_type(fp, type_str, sizeof type_str);
        if (tp > 0 && ndecl < MAXDECL) {
          struct Decl fd;
          memset(&fd, 0, sizeof fd);
          strcpy(fd.type, type_str);
          int dp = parse_declarator(tp, &fd);
          if (dp > 0) {
            fd.start = fp;
            fd.end = dp; /* at ; */
            fd.is_for = 1;
            decls[ndecl++] = fd;
          }
        }
        p = paren_end;
      }
    } else if (strncmp(src + p, "if", 2) == 0 && !is_ident_char(src[p + 2])) {
      p += 2;
      p = skipws(p);
      if (src[p] == '(') p = skip_balanced(p);
    } else if (strncmp(src + p, "while", 5) == 0 &&
               !is_ident_char(src[p + 5])) {
      p += 5;
      p = skipws(p);
      if (src[p] == '(') p = skip_balanced(p);
    } else if (src[p] == '#') {
      /* skip preprocessor directive */
      while (p < body_end && src[p] != '\n') p++;
      if (p < body_end) p++;
    } else {
      /* skip to end of statement */
      int p0 = p;
      while (p < body_end && src[p] != ';' && src[p] != '{' && src[p] != '}' &&
             src[p] != '#') {
        if (src[p] == '(' || src[p] == '[') p = skip_balanced(p);
        else
          p++;
      }
      if (p < body_end && src[p] == ';') p++;
      if (p == p0) p++; /* safety: always advance */
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
      if (cmp == 0) {
        int ki = (decls[ci].full_name[0] == '*') ? 0 : strchr(decls[ci].full_name, '[') ? 2 : 1;
        int kj = (decls[cj].full_name[0] == '*') ? 0 : strchr(decls[cj].full_name, '[') ? 2 : 1;
        cmp = ki - kj; /* pointers, then scalars, then arrays */
      }
      if (cmp == 0) cmp = strcmp(decls[ci].name, decls[cj].name);
      if (cmp > 0) {
        int t = order[i];
        order[i] = order[j];
        order[j] = t;
      }
    }

  /* emit { */
  emit("{\n", 2);

  /* emit grouped declarations: separate line for pointers vs scalars */
  char prev_type[128] = "";
  int prev_ptr = -1;
  int line_start = 1;
  for (int oi = 0; oi < ndecl; oi++) {
    int i = order[oi];
    if (!keep[i]) continue;
    int kind = (decls[i].full_name[0] == '*') ? 0 : strchr(decls[i].full_name, '[') ? 2 : 1;
    if (strcmp(decls[i].type, prev_type) != 0 || kind != prev_ptr) {
      if (!line_start) emit(";\n", 2);
      fprintf(outfp, "  %s %s", decls[i].type, decls[i].full_name);
      strcpy(prev_type, decls[i].type);
      prev_ptr = kind;
      line_start = 0;
    } else {
      fprintf(outfp, ", %s", decls[i].full_name);
    }
  }
  if (!line_start) emit(";\n", 2);

  /* emit body, replacing declarations with assignments */
  p = f->body_start + 1;
  /* skip whitespace before first real content */
  while (p < f->body_end - 1 && isspace((unsigned char)src[p])) p++;
  int need_sep = 1; /* emit one blank line after declarations */
  while (p < f->body_end - 1) {
    /* check if current position matches a collected declaration */
    int found = -1;
    for (int i = 0; i < ndecl; i++) {
      if (decls[i].start == p) {
        found = i;
        break;
      }
    }
    if (found >= 0) {
      struct Decl *d = &decls[found];
      if (d->is_for) {
        if (d->has_init) {
          int eq = d->start;
          while (eq < d->val_end && src[eq] != '=') eq++;
          fprintf(outfp, "%s ", d->name);
          emit(src + eq, d->val_end - eq);
        }
        p = d->end;
      } else {
        /* find indentation from original line */
        int ls = d->start, any_init = 0;
        while (ls > 0 && src[ls - 1] != '\n') ls--;
        /* emit all declarators sharing this start */
        for (int di = 0; di < ndecl; di++) {
          if (decls[di].start != d->start || !decls[di].has_init) continue;
          if (need_sep) { emit("\n", 1); need_sep = 0; }
          int il;
          for (il = ls; il < d->start && isspace((unsigned char)src[il]); il++)
            fputc(src[il], outfp);
          int eq = decls[di].val_end - 1;
          while (eq > d->start && src[eq] != '=') eq--;
          fprintf(outfp, "%s ", decls[di].name);
          emit(src + eq, decls[di].val_end - eq);
          emit(";\n", 2);
          any_init = 1;
        }
        p = d->end;
        if (!any_init)
          while (p < f->body_end - 1 && isspace((unsigned char)src[p])) p++;
      }
    } else {
      if (need_sep) {
        emit("\n", 1);
        /* restore indentation from original source */
        int ls = p;
        while (ls > 0 && src[ls - 1] != '\n') ls--;
        while (ls < p && isspace((unsigned char)src[ls]))
          fputc(src[ls++], outfp);
        need_sep = 0;
      }
      fputc(src[p], outfp);
      p++;
    }
  }

  /* emit } */
  emit("}\n", 2);
}

int main(int argc, char **argv) {
  char *only_func = NULL;
  char *filename = NULL;
  int list_mode = 0;
  int inplace = 0;
  int i;

  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0) {
      fprintf(stderr, "usage: cdecl [-f func] [-l] [-i] [file]\n");
      return 0;
    } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) only_func = argv[++i];
    else if (strcmp(argv[i], "-l") == 0)
      list_mode = 1;
    else if (strcmp(argv[i], "-i") == 0)
      inplace = 1;
    else if (argv[i][0] != '-')
      filename = argv[i];
  }

  if (filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
      fprintf(stderr, "cdecl: cannot open %s\n", filename);
      return 1;
    }
    srcn = fread(src, 1, MAXSRC - 1, fp);
    fclose(fp);
  } else {
    srcn = fread(src, 1, MAXSRC - 1, stdin);
  }
  src[srcn] = 0;

  find_functions();

  if (list_mode) {
    for (i = 0; i < nfuncs; i++)
      fprintf(stdout, "%4d  %s\n", funcs[i].body_end - funcs[i].body_start,
              funcs[i].name);
    return 0;
  }

  if (inplace && !filename) {
    fprintf(stderr, "cdecl: -i requires a file argument\n");
    return 1;
  }

  char tmppath[512] = "";
  outfp = stdout;
  if (inplace) {
    snprintf(tmppath, sizeof tmppath, "/tmp/cdecl.%d.tmp", (int)getpid());
    outfp = fopen(tmppath, "w");
    if (!outfp) {
      fprintf(stderr, "cdecl: cannot write %s\n", tmppath);
      return 1;
    }
  }

  if (only_func) {
    int last = 0;
    for (i = 0; i < nfuncs; i++) {
      if (strcmp(funcs[i].name, only_func) != 0) continue;
      fwrite(src + last, 1, funcs[i].body_start - last, outfp);
      process_function(&funcs[i]);
      last = funcs[i].body_end;
    }
    fwrite(src + last, 1, srcn - last, outfp);
  } else {
    int last = 0;
    for (i = 0; i < nfuncs; i++) {
      fwrite(src + last, 1, funcs[i].body_start - last, outfp);
      process_function(&funcs[i]);
      last = funcs[i].body_end;
    }
    fwrite(src + last, 1, srcn - last, outfp);
  }

  if (inplace && filename) {
    fclose(outfp);
    rename(tmppath, filename);
  }
  return 0;
}
