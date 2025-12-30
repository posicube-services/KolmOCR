<!-- bbox: [48,37,952,111] -->
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TITLE 100
#define MAX_CONTENT 1000
#define DATA_FILE "memos.txt"
```

<!-- bbox: [48,113,952,169] -->
```c
typedef struct {
    int id;
    char title[MAX_TITLE];
    char content[MAX_CONTENT];
} Memo;
```

<!-- bbox: [48,172,952,219] -->
```c
void print_menu(){
    printf("\n==== Simple Text Notepad ====\n");
    printf("1.List 2.Add 3.View 4.Delete 0.Exit\n> ");
}
```

<!-- bbox: [48,222,952,331] -->
```c
void list_memos(){
    FILE *fp=fopen(DATA_FILE,"r");
    Memo m; int found=0;
    if(!fp){printf("No memos.\n");return;}
    while(fread(&m,sizeof(Memo),1,fp)){
        printf("ID:%d | %s\n",m.id,m.title);
        found=1;
    }
    if(!found) printf("Empty.\n");
    fclose(fp);
}
```

<!-- bbox: [48,334,952,425] -->
```c
int get_next_id(){
    FILE *fp=fopen(DATA_FILE,"r");
    Memo m; int max=0;
    if(!fp) return 1;
    while(fread(&m,sizeof(Memo),1,fp))
        if(m.id>max) max=m.id;
    fclose(fp);
    return max+1;
}
```

<!-- bbox: [48,428,952,581] -->
```c
void add_memo(){
    Memo m; char line[256];
    m.id=get_next_id();
    printf("Title: ");
    fgets(m.title,MAX_TITLE,stdin);
    m.title[strcspn(m.title,"\n")]=0;
    m.content[0]=0;
    printf("Content ('.' to end):\n");
    while(fgets(line,sizeof(line),stdin)){
        if(strcmp(line,".\n")==0) break;
        if(strlen(m.content)+strlen(line)<MAX_CONTENT)
            strcat(m.content,line);
    }
    FILE *fp=fopen(DATA_FILE,"ab");
    if(fp){fwrite(&m,sizeof(Memo),1,fp);fclose(fp);}
}
```

<!-- bbox: [48,585,952,703] -->
```c
void view_memo(){
    int id; Memo m;
    printf("ID: ");
    if(scanf("%d",&id)!=1){while(getchar()!='\n');return;}
    while(getchar()!='\n');
    FILE *fp=fopen(DATA_FILE,"r");
    if(!fp) return;
    while(fread(&m,sizeof(Memo),1,fp))
        if(m.id==id)
            printf("[%d] %s\n%s\n",m.id,m.title,m.content);
    fclose(fp);
}
```

<!-- bbox: [48,705,952,823] -->
```c
void delete_memo(){
    int id; Memo m;
    printf("ID: ");
    if(scanf("%d",&id)!=1){while(getchar()!='\n');return;}
    while(getchar()!='\n');
    FILE *fp=fopen(DATA_FILE,"r");
    FILE *tmp=fopen("tmp.txt","wb");
    while(fread(&m,sizeof(Memo),1,fp))
        if(m.id!=id) fwrite(&m,sizeof(Memo),1,tmp);
    fclose(fp); fclose(tmp);
    remove(DATA_FILE); rename("tmp.txt",DATA_FILE);
}
```

<!-- bbox: [48,826,952,962] -->
```c
int main(){
    int c;
    for(;;){
        print_menu();
        if(scanf("%d",&c)!=1){while(getchar()!='\n');continue;}
        while(getchar()!='\n');
        if(c==1) list_memos();
        else if(c==2) add_memo();
        else if(c==3) view_memo();
        else if(c==4) delete_memo();
        else if(c==0) break;
    }
    return 0;
}
```
