package iialib.games.contest;

import java.util.ArrayList;

public interface IRule {
    IRule play(String move, String role);

    ArrayList<String> possibleMoves(String role);

    ArrayList<IRule> successors(String role);

    boolean isValidMove(String move, String role);

    boolean isGameOver();

    boolean isWinner(String role);

    boolean isTie();

    String getFirstRole();

    String getSecondRole();

    int getTimeout();

    int getTotalTimeout();

    int[][] getBoard();
}
