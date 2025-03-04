package iialib.games.algs.algorithms;

import iialib.games.algs.GameAlgorithm;
import iialib.games.model.IBoard;
import iialib.games.model.IMove;
import iialib.games.model.IRole;

import java.sql.SQLOutput;
import java.util.ArrayList;
import java.util.Random;

public class RandomChoice<Move extends IMove, Role extends IRole, Board extends IBoard<Move, Role, Board>> implements GameAlgorithm<Move, Role, Board> {

    // Attributes
    private final Role playerMaxRole;
    private final Random randomgen = new Random();

    // --------- Constructors ---------

    public RandomChoice(Role playerMaxRole) {
        this.playerMaxRole = playerMaxRole;
    }

    @Override
    public Move bestMove(Board board, Role playerRole) {
        System.out.println("[Random]");
        System.out.println(board);

        // Compute all possible moves for maxPlayer
        ArrayList<Move> allMoves = board.possibleMoves(playerMaxRole);
        //System.out.println("* " + allMoves.size() + " possible moves");
        Move bestMove = allMoves.get(randomgen.nextInt(allMoves.size()));

        return bestMove;
    }

}
