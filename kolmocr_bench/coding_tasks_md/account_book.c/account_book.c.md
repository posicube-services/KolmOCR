<!-- bbox: [35,17,707,965] -->
c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_CAT 50
#define DATA_FILE "account_book.txt"
typedef struct {
    char type; // 'I' income, 'E' expense
    char category[MAX_CAT];
    int amount;
} Record;
void print_menu() {
    printf("\n==== Simple Account Book ====\n");
    printf("1. Add record\n");
    printf("2. List records\n");
    printf("3. Summary\n");
    printf("0. Exit\n");
    printf("Select: ");
}
void add_record() {
    Record r;
    char line[128];
    printf("Type (I=income, E=expense): ");
    fgets(line, sizeof(line), stdin);
    r.type = line[0] == 'E' || line[0] == 'e' ? 'E' : 'I';
    printf("Category: ");
    fgets(r.category, MAX_CAT, stdin);
    if (r.category[strlen(r.category) - 1] == '\n')
        r.category[strlen(r.category) - 1] = '\0';
    printf("Amount: ");
    fgets(line, sizeof(line), stdin);
    r.amount = atoi(line);
    FILE *fp = fopen(DATA_FILE, "a");
    if (!fp) {
        printf("Failed to open file.\n");
        return;
    }
    fprintf(fp, "%c,%s,%d\n", r.type, r.category, r.amount);
    fclose(fp);
    printf("Record added.\n");
}
void list_records() {
    FILE *fp = fopen(DATA_FILE, "r");
    if (!fp) {
        printf("No records.\n");
        return;
    }
    printf("\n[Records]\n");
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        char type, category[MAX_CAT];
        int amount;
        if (sscanf(line, "%c,%49[^,],%d", &type, category, &amount) == 3) {
            printf("%s: %-10s %d\n", type == 'I' ? "Income " : "Expense", category, amount);
        }
    }
    fclose(fp);
}
void summary() {
    FILE *fp = fopen(DATA_FILE, "r");
    if (!fp) {
        printf("No records.\n");
        return;
    }
    int total_income = 0, total_expense = 0;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        char type, category[MAX_CAT];
        int amount;
        if (sscanf(line, "%c,%49[^,],%d", &type, category, &amount) == 3) {
            if (type == 'I') total_income += amount;
            else total_expense += amount;
        }
    }
    fclose(fp);
    printf("\n[Summary]\n");
    printf("Total income : %d\n", total_income);
    printf("Total expense: %d\n", total_expense);
    printf("Balance      : %d\n", total_income - total_expense);
}
int main() {
    char line[16];
    while (1) {
        print_menu();
        if (!fgets(line, sizeof(line), stdin)) break;
        int choice = atoi(line);
        switch (choice) {
            case 1: add_record(); break;
            case 2: list_records(); break;
            case 3: summary(); break;
            case 0: printf("Bye.\n"); return 0;
            default: printf("Unknown menu.\n");
        }
    }
    return 0;
}
