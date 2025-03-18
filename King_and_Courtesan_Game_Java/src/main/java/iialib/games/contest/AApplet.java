package iialib.games.contest;

import javax.swing.*;
import java.awt.*;

public abstract class AApplet extends JApplet {
    public abstract Dimension getDimension();

    public abstract void buildUI(Container contentPane);

    public abstract void setMyFrame(JFrame f);

    public abstract void addBoard(String string, int[][] is);

    public abstract void update(Graphics graphics, Insets insets);
}
