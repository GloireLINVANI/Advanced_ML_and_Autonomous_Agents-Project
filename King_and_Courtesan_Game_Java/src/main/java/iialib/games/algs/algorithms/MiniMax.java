package iialib.games.algs.algorithms;

import iialib.games.algs.GameAlgorithm;
import iialib.games.algs.IHeuristic;
import iialib.games.model.IBoard;
import iialib.games.model.IMove;
import iialib.games.model.IRole;

import java.util.ArrayList;

public class MiniMax<Move extends IMove, Role extends IRole, Board extends IBoard<Move, Role, Board>> implements GameAlgorithm<Move, Role, Board> {

    // Constants
    /**
     * Default value for depth limit
     */
    private final static int DEPTH_MAX_DEFAUT = 6;

    // Attributes
    /**
     * Role of the max player
     */
    private final Role playerMaxRole;
    /**
     * Role of the min player
     */
    private final Role playerMinRole;
    /**
     * Heuristic used by the max player
     */
    private final IHeuristic<Board, Role> h;
    /**
     * Algorithm max depth
     */
    private int depthMax = DEPTH_MAX_DEFAUT;
    /**
     * Number of internal visited (developed) nodes (for stats)
     */
    private int nbNodes;
    /**
     * Number of leaves nodes nodes (for stats)
     */
    private int nbLeaves;


    // --------- CONSTRUCTORS ---------

    public MiniMax(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h) {
        this.playerMaxRole = playerMaxRole;
        this.playerMinRole = playerMinRole;
        this.h = h;
    }

    public MiniMax(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h, int depthMax) {
        this(playerMaxRole, playerMinRole, h);
        this.depthMax = depthMax;
    }

    // --------- IAlgo METHODS ---------

    @Override
    public Move bestMove(Board board, Role playerRole) {
        // In this version, the playerRole parameter is ignored,
        // since playerMaxRole is fixed in the constructor.
        return bestMove(board);
    }

    public Move bestMove(Board board) {
        System.out.println("[MiniMax]");
        ArrayList<Move> coupsPossibles = board.possibleMoves(playerMaxRole);
        nbNodes++;
        Move bestMove = coupsPossibles.get(0);
        int Max = minMax(board.play(bestMove, playerMaxRole), 1);
        for (int i = 1; i < coupsPossibles.size(); i++) {
            Move move = coupsPossibles.get(i);
            int moveValue = minMax(board.play(move, playerMaxRole), 1);
            if (moveValue > Max) {
                Max = moveValue;
                bestMove = move;
            }
        }
        return bestMove;
    }

    // --------- PUBLIC METHODS ---------

    public String toString() {
        return "MiniMax(ProfMax=" + depthMax + ")";
    }

    // --------- PRIVATE METHODS ---------
    private int maxMin(Board board, int profondeur) {
        if (board.isGameOver() || profondeur == depthMax) {
            nbLeaves++;
            return this.h.eval(board, playerMaxRole);
        } else {
            nbNodes++;
            int Max = IHeuristic.MIN_VALUE;
            ArrayList<Move> coupsPossibles = board.possibleMoves(playerMaxRole);
            for (Move coupsPossible : coupsPossibles) {
                Max = Math.max(Max, minMax(board.play(coupsPossible, playerMaxRole), profondeur + 1));
            }
            return Max;
        }
    }

    private int minMax(Board board, int profondeur) {
        if (board.isGameOver() || profondeur == depthMax) {
            nbLeaves++;
            return this.h.eval(board, playerMaxRole);
        } else {
            nbNodes++;
            int Min = IHeuristic.MAX_VALUE;
            ArrayList<Move> coupsPossibles = board.possibleMoves(playerMinRole);
            for (Move coupsPossible : coupsPossibles) {
                Min = Math.min(Min, maxMin(board.play(coupsPossible, playerMinRole), profondeur + 1));
            }
            return Min;
        }
    }

    public void stats() {
        System.out.println(playerMaxRole + " Number of nodes : " + nbNodes);
        System.out.println(playerMaxRole + " Number of leaves : " + nbLeaves);
    }

}
