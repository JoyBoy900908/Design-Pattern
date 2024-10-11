import java.util.ArrayList;
import java.util.List;

abstract class Component {
    public void add(Component component) {
        throw new UnsupportedOperationException();
    }

    public void remove(Component component) {
        throw new UnsupportedOperationException();
    }

    public abstract void operation();
}

class Leaf extends Component {
    @Override
    public void operation() {
        System.out.println("Leaf operation");
    }
}

class Composite extends Component {
    private List<Component> children = new ArrayList<>();

    @Override
    public void add(Component component) {
        children.add(component);
    }

    @Override
    public void remove(Component component) {
        children.remove(component);
    }

    @Override
    public void operation() {
        for (Component child : children) {
            child.operation();
        }
    }
}

public class Main {
    public static void main(String[] args) {
        Leaf leaf1 = new Leaf();
        Leaf leaf2 = new Leaf();
        Composite composite = new Composite();

        composite.add(leaf1);
        composite.add(leaf2);

        // Correct usage: leaf does not support add operation
        // leaf1.add(new Leaf()); // This line should be removed
        leaf1.operation();

        composite.operation();
    }
}
