package games.kac;

import iialib.games.model.IMove;

public class KingAndCourtesanMove implements IMove {
    public final char fromLig;
    public final int fromCol;
    public final char toLig;
    public final int toCol;

    public KingAndCourtesanMove(char fromLig, int fromCol, char toLig, int toCol) {
        this.fromLig = fromLig;
        this.fromCol = fromCol;
        this.toLig = toLig;
        this.toCol = toCol;
    }

    public KingAndCourtesanMove(String move) {
        String[] helper = move.split("-");
        this.fromLig = helper[0].charAt(0);
        this.fromCol = Integer.parseInt(String.valueOf(helper[0].charAt(1)));
        this.toLig = helper[1].charAt(0);
        this.toCol = Integer.parseInt(String.valueOf(helper[1].charAt(1)));
    }

    public char getFromLig() {
        return fromLig;
    }

    public int getFromCol() {
        return fromCol;
    }

    public char getToLig() {
        return toLig;
    }

    public int getToCol() {
        return toCol;
    }

    @Override
    public String toString() {
        return "" + this.fromLig + this.fromCol + "-" + this.toLig + this.toCol;
    }
}
