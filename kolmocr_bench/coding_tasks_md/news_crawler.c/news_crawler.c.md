<!-- bbox: [35,33,624,934] -->
c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CMD_BUF 512
#define LINE_BUF 2048
#define TMP_FILE "page.html"
void fetch_page(const char *url) {
    char cmd[CMD_BUF];
    snprintf(cmd, sizeof(cmd), "curl -s \"%s\" -o %s", url, TMP_FILE);
    int ret = system(cmd);
    if (ret != 0) {
        printf("Failed to fetch page. Make sure curl is installed.\n");
    }
}
void extract_titles() {
    FILE *fp = fopen(TMP_FILE, "r");
    if (!fp) {
        printf("No page downloaded.\n");
        return;
    }
    char line[LINE_BUF];
    int count = 0;
    printf("\n[Lines containing <title> or <h1>..</h1>]\n");
    while (fgets(line, sizeof(line), fp) && count < 20) {
        if (strstr(line, "<title>") || strstr(line, "<h1")) {
            printf("%s", line);
            count++;
        }
    }
    if (count == 0) {
        printf("No titles found (simple parser).\n");
    }
    fclose(fp);
}
int main() {
    char url[256];
    printf("Enter news site URL (e.g. https://news.ycombinator.com):\n> ");
    if (!fgets(url, sizeof(url), stdin)) return 1;
    if (url[strlen(url) - 1] == '\n') url[strlen(url) - 1] = '\0';
    if (strlen(url) == 0) {
        printf("URL required.\n");
        return 1;
    }
    fetch_page(url);
    extract_titles();
    printf("\nDone.\n");
    return 0;
}
