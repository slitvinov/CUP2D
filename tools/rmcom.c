
#include <stdio.h>

int main(void) {
  int c, d, in_str = 0, in_chr = 0, esc = 0;
  while ((c = getchar()) != EOF) {
    if (esc) {
      putchar(c);
      esc = 0;
      continue;
    }
    if (in_str) {
      putchar(c);
      if (c == '\\') esc = 1;
      else if (c == '"')
        in_str = 0;
      continue;
    }
    if (in_chr) {
      putchar(c);
      if (c == '\\') esc = 1;
      else if (c == '\'')
        in_chr = 0;
      continue;
    }
    if (c == '"') {
      putchar(c);
      in_str = 1;
      continue;
    }
    if (c == '\'') {
      putchar(c);
      in_chr = 1;
      continue;
    }
    if (c == '/') {
      d = getchar();
      if (d == '/') {
        while ((d = getchar()) != EOF && d != '\n');
        if (d == '\n') putchar('\n');
      } else if (d == '*') {
        for (;;) {
          c = getchar();
          if (c == EOF) return 0;
          if (c == '*') {
            d = getchar();
            if (d == '/') break;
            if (d == EOF) return 0;
          }
        }
        putchar(' ');
      } else {
        putchar(c);
        if (d != EOF) ungetc(d, stdin);
      }
      continue;
    }
    putchar(c);
  }
  return 0;
}
