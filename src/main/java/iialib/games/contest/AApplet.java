package iialib.games.contest;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Insets;

import javax.swing.JApplet;
import javax.swing.JFrame;

import iialib.games.model.IBoard;

public abstract class AApplet extends JApplet {
	public abstract Dimension getDimension();
	public abstract void buildUI(Container contentPane);
	public abstract void setMyFrame(JFrame f);
	public abstract void addBoard(String string, int[][] is);
	public abstract void update(Graphics graphics, Insets insets);
}
