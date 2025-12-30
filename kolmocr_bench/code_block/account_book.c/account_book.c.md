<!-- bbox: [61,42,938,86] -->
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
```

<!-- bbox: [61,88,938,118] -->
```c
#define MAX_CAT 50
#define DATA_FILE "account_book.txt"
```

<!-- bbox: [61,119,938,193] -->
```c
typedef struct {
    char type;       
    char category[MAX_CAT];
    int amount;
} Record;
```

<!-- bbox: [61,195,938,254] -->
```c
void print_menu() {
    printf("\n==== Account Book ====\n");
    printf("1. Add\n2. List\n3. Summary\n0. Exit\n> ");
}
```

<!-- bbox: [61,256,938,567] -->
```c
void add_record() {
    Record r;
    char buf[64];

    printf("Type (I/E): ");
    fgets(buf, sizeof(buf), stdin);
    r.type = (buf[0] == 'E') ? 'E' : 'I';

    printf("Category: ");
    fgets(r.category, MAX_CAT, stdin);
    r.category[strcspn(r.category, "\n")] = 0;

    printf("Amount: ");
    fgets(buf, sizeof(buf), stdin);
    r.amount = atoi(buf);

    FILE *fp = fopen(DATA_FILE, "a");
    if (!fp) return;
    fprintf(fp, "%c,%s,%d\n", r.type, r.category, r.amount);
    fclose(fp);
}
```

<!-- bbox: [61,568,938,775] -->
```c
void summary() {
    FILE *fp = fopen(DATA_FILE, "r");
    int in = 0, out = 0;
    char line[128], cat[MAX_CAT], t;
    int amt;

    if (!fp) return;
    while (fgets(line, sizeof(line), fp))
        if (sscanf(line, "%c,%49[^,],%d", &t, cat, &amt) == 3)
            (t == 'I') ? (in += amt) : (out += amt);

    fclose(fp);
    printf("Income:%d  Expense:%d  Balance:%d\n", in, out, in - out);
}
```

<!-- bbox: [61,777,938,955] -->
```c
int main() {
    char buf[8];
    while (1) {
        print_menu();
        fgets(buf, sizeof(buf), stdin);
        switch (atoi(buf)) {
            case 1: add_record(); break;
            case 3: summary(); break;
            case 0: return 0;
        }
    }
}
```
