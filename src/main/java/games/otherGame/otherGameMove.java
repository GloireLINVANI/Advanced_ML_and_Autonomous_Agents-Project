package games.otherGame;

import iialib.games.model.IMove;

public class otherGameMove implements IMove {

    public final int x;
    public final int y;

    otherGameMove(int x, int y) {
        this.x = x;
        this.y = y;
    }

    @Override
    public String toString() {
        return "Move{" + x + "," + y + "}";
    }
}
