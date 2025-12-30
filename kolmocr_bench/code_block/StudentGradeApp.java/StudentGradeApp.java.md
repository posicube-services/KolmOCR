<!-- bbox: [48,34,952,61] -->
```java
import java.io.*; import java.util.*;
```

<!-- bbox: [48,64,952,181] -->
```java
class S implements Serializable{
  String n; Map<String,Integer> m=new LinkedHashMap<>();
  S(String n){this.n=n;}
  double a(){ if(m.isEmpty()) return 0;
    int s=0; for(int v:m.values()) s+=v; return s/(double)m.size(); }
}
```

<!-- bbox: [48,183,952,515] -->
```java
public class G{
  static final String F="students.ser";
  List<S> a=new ArrayList<>(); Scanner sc=new Scanner(System.in);
  public static void main(String[] args){ new G().go(); }

  void go(){ load();
    for(;;){
      System.out.print("\\n1.L 2.A 3.S 4.R 0.X > ");
      switch(sc.nextLine().trim()){
        case "1" -> list();
        case "2" -> add();
        case "3" -> score();
        case "4" -> rank();
        case "0" -> { save(); return; }
        default  -> System.out.println("?");
      }
    }
  }
```

<!-- bbox: [48,518,952,653] -->
```java
S f(String n){ for(S s:a) if(s.n.equalsIgnoreCase(n)) return s; return null; }

  void add(){
    System.out.print("Name: "); String n=sc.nextLine().trim();
    if(n.isEmpty()||f(n)!=null){ System.out.println("No/dup"); return; }
    a.add(new S(n)); System.out.println("OK");
  }
```

<!-- bbox: [48,655,952,807] -->
```java
void score(){
    System.out.print("Name: "); S s=f(sc.nextLine().trim());
    if(s==null){ System.out.println("NF"); return; }
    System.out.print("Sub: "); String sub=sc.nextLine().trim();
    System.out.print("Score: ");
    try{ s.m.put(sub,Integer.parseInt(sc.nextLine().trim())); }
    catch(Exception e){ System.out.println("Bad"); }
  }
```

<!-- bbox: [48,811,952,963] -->
```java
void list(){
    if(a.isEmpty()){ System.out.println("(empty)"); return; }
    for(S s:a){
      System.out.printf("%s(%.2f)\\n", s.n, s.a());
      for(var e:s.m.entrySet())
        System.out.println(" "+e.getKey()+":"+e.getValue());
    }
  }
```
