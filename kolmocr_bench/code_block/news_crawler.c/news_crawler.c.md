<!-- bbox: [48,40,952,205] -->
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CMD_BUF 512
#define LINE_BUF 2048
#define TMP_FILE "page.html"
```

<!-- bbox: [48,212,952,377] -->
```c
void fetch_page(const char *url) {
    char cmd[CMD_BUF];
    snprintf(cmd, sizeof(cmd),
             "curl -s \"%s\" -o %s", url, TMP_FILE);
    if (system(cmd) != 0)
        printf("Fetch failed.\n");
}
```

<!-- bbox: [48,383,952,686] -->
```c
void extract_titles() {
    FILE *fp = fopen(TMP_FILE, "r");
    if (!fp) return;

    char line[LINE_BUF];
    int count = 0;
    while (fgets(line, sizeof(line), fp) && count < 20) {
        if (strstr(line, "<title") || strstr(line, "<h1")) {
            printf("%s", line);
            count++;
        }
    }
    fclose(fp);
}
```

<!-- bbox: [48,693,952,957] -->
```c
int main() {
    char url[256];
    printf("URL > ");
    if (!fgets(url, sizeof(url), stdin)) return 1;
    url[strcspn(url, "\n")] = 0;

    if (!*url) return 1;

    fetch_page(url);
    extract_titles();
    return 0;
}
```
