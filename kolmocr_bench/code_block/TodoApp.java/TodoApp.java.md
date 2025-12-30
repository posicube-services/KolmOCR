<!-- bbox: [48,34,952,80] -->
```java
import java.io.*;
import java.util.*;
```

<!-- bbox: [48,82,952,183] -->
```java
class Todo implements Serializable{
  int id; String t; boolean d;
  Todo(int id,String t){this.id=id; this.t=t;}
  public String toString(){return "["+id+"] "+(d?"[x] ":"[ ] ")+t;}
}
```

<!-- bbox: [48,186,952,287] -->
```java
public class TodoApp{
  static final String F="todos.ser";
  List<Todo> a=new ArrayList<>(); int next=1;
  Scanner sc=new Scanner(System.in);
  public static void main(String[] args){new TodoApp().go();}
```

<!-- bbox: [48,290,952,555] -->
```java
void go(){ load();
    for(;;){
      System.out.print("\\n1.L 2.A 3.T 4.R 0.X > ");
      switch(sc.nextLine().trim()){
        case "1" -> list();
        case "2" -> add();
        case "3" -> toggle();
        case "4" -> remove();
        case "0" -> { save(); return; }
        default -> System.out.println("?");
      }
    }
  }
  int n(){ try{ return Integer.parseInt(sc.nextLine().trim()); }catch(Exception e){ return -1; } }
```

<!-- bbox: [48,558,952,750] -->
```java
void list(){ if(a.isEmpty()) System.out.println("(empty)");
    else for(Todo t:a) System.out.println(t); }
  void add(){ System.out.print("Title: ");
    String t=sc.nextLine().trim(); if(t.isEmpty()) return;
    a.add(new Todo(next++, t)); }
  void toggle(){ System.out.print("ID: "); int id=n();
    for(Todo t:a) if(t.id==id){ t.d=!t.d; return; }
    System.out.println("Not found."); }
  void remove(){ System.out.print("ID: "); int id=n();
    a.removeIf(t -> t.id==id); }
```

<!-- bbox: [48,753,952,963] -->
```java
@SuppressWarnings("unchecked")
  void load(){ File f=new File(F); if(!f.exists()) return;
    try(ObjectInputStream in=new ObjectInputStream(new FileInputStream(f))){
      a=(List<Todo>)in.readObject();
      for(Todo t:a) if(t.id>=next) next=t.id+1;
    }catch(Exception e){ System.out.println("Load fail"); } }
  void save(){
    try(ObjectOutputStream out=new ObjectOutputStream(new FileOutputStream(F))){
      out.writeObject(a);
    }catch(IOException e){ System.out.println("Save fail"); } }
}
```
