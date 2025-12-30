<!-- bbox: [57,39,942,58] -->
```java
import java.io.*;
import java.util.*;
```

<!-- bbox: [57,60,942,108] -->
```java
class Account implements Serializable {
  private static final long serialVersionUID = 1L;
  String owner;
  int balance;
  List<String> history = new ArrayList<>();
```

<!-- bbox: [57,109,942,148] -->
```java
Account(String owner){
    this.owner = (owner == null || owner.isBlank()) ? "Unknown" : owner.trim();
    history.add("Account created");
  }
```

<!-- bbox: [57,149,942,188] -->
```java
void deposit(int amount){
    balance += amount;
    history.add("Deposit: " + amount + " (bal=" + balance + ")");
  }
```

<!-- bbox: [57,189,942,256] -->
```java
boolean withdraw(int amount){
    if (amount > balance) return false;
    balance -= amount;
    history.add("Withdraw: " + amount + " (bal=" + balance + ")");
    return true;
  }
}
```

<!-- bbox: [57,257,942,296] -->
```java
public class BankApp {
  private static final String DATA_FILE = "account.ser";
  private Account account;
  private final Scanner sc = new Scanner(System.in);
```

<!-- bbox: [57,297,942,345] -->
```java
public static void main(String[] args){
    BankApp app = new BankApp();
    app.load();
    app.run();
  }
```

<!-- bbox: [57,346,942,500] -->
```java
private void run(){
    if (account == null){
      System.out.print("Owner name: ");
      account = new Account(sc.nextLine());
    }
    for(;;){
      menu();
      switch(sc.nextLine().trim()){
        case "1" -> info();
        case "2" -> deposit();
        case "3" -> withdraw();
        case "0" -> { save(); System.out.println("Bye."); return; }
        default  -> System.out.println("Unknown.");
      }
    }
  }
```

<!-- bbox: [57,502,942,550] -->
```java
private void menu(){
    System.out.println("\\n== Mini Bank ==");
    System.out.println("1.Info  2.Deposit  3.Withdraw  0.Exit");
    System.out.print("> ");
  }
```

<!-- bbox: [57,551,942,590] -->
```java
private void info(){
    System.out.println("Owner: " + account.owner);
    System.out.println("Bal  : " + account.balance);
  }
```

<!-- bbox: [57,591,942,658] -->
```java
private void deposit(){
    System.out.print("Deposit amount: ");
    int x = parsePositive(sc.nextLine());
    if (x <= 0) return;
    account.deposit(x);
    System.out.println("OK");
  }
```

<!-- bbox: [57,659,942,717] -->
```java
private void withdraw(){
    System.out.print("Withdraw amount: ");
    int x = parsePositive(sc.nextLine());
    if (x <= 0) return;
    System.out.println(account.withdraw(x) ? "OK" : "Not enough");
  }
```

<!-- bbox: [57,718,942,814] -->
```java
private int parsePositive(String s){
    try{
      int x = Integer.parseInt(s.trim());
      if (x <= 0) System.out.println("Must be positive.");
      return x;
    }catch(Exception e){
      System.out.println("Invalid number.");
      return -1;
    }
  }
```

<!-- bbox: [57,816,942,912] -->
```java
private void load(){
    File f = new File(DATA_FILE);
    if (!f.exists()) return;
    try(ObjectInputStream in = new ObjectInputStream(new FileInputStream(f))){
      account = (Account) in.readObject();
      System.out.println("Loaded: " + account.owner);
    }catch(Exception e){
      System.out.println("Load failed: " + e.getMessage());
    }
  }
```
