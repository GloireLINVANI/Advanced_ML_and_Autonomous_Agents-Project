package iialib.games.algs.algorithms;

import iialib.games.algs.GameAlgorithm;
import iialib.games.algs.IHeuristic;
import iialib.games.model.IBoard;
import iialib.games.model.IMove;
import iialib.games.model.IRole;

import java.util.ArrayList;

public class AlphaBeta<Move extends IMove, Role extends IRole, Board extends IBoard<Move, Role, Board>> implements GameAlgorithm<Move, Role, Board> {
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
     * Algorithm max depth
     */
    private int depthMax = DEPTH_MAX_DEFAUT;
    /**
     * Heuristic used by the max player
     */
    private final IHeuristic<Board, Role> h;

    public AlphaBeta(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h) {
        this.playerMaxRole = playerMaxRole;
        this.playerMinRole = playerMinRole;
        this.h = h;
    }

    public AlphaBeta(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h, int depthMax) {
        this(playerMaxRole, playerMinRole, h);
        this.depthMax = depthMax;
    }

    // --------- IAlgo METHODS ---------

    @Override
    public Move bestMove(Board board, Role playerRole) {
        ArrayList<Move> coupsPossibles = board.possibleMoves(playerMaxRole);
        Move bestMove = null;
        int alpha = h.MIN_VALUE;
        int beta = h.MAX_VALUE;
        for (Move move : coupsPossibles) {
            int moveValue = minMax(board.play(move, playerMaxRole), 1, alpha, beta);
            if (moveValue > alpha) {
                alpha = moveValue;
                bestMove = move;
            }
        }
        return bestMove;
    }

    // --------- PRIVATE METHODS ---------
    private int maxMin(Board board, int profondeur, int alpha, int beta) {
        if (board.isGameOver() || profondeur == depthMax) {
            return h.eval(board, playerMaxRole);
        } else {
            ArrayList<Move> coupsPossibles = board.possibleMoves(playerMaxRole);
            for (Move coupsPossible : coupsPossibles) {
                int moveValue = minMax(board.play(coupsPossible, playerMaxRole), profondeur + 1, alpha, beta);
                alpha = Math.max(alpha, moveValue);
                if (alpha >= beta) {
                    return beta;
                }
            }
            return alpha;
        }
    }

    private int minMax(Board board, int profondeur, int alpha, int beta) {
        if (board.isGameOver() || profondeur == depthMax) {
            return this.h.eval(board, playerMaxRole);
        } else {
            ArrayList<Move> coupsPossibles = board.possibleMoves(playerMinRole);
            for (Move coupsPossible : coupsPossibles) {
                int moveValue = maxMin(board.play(coupsPossible, playerMinRole), profondeur + 1, alpha, beta);
                beta = Math.min(beta, moveValue);
                if (beta <= alpha) {
                    return alpha;
                }
            }
            return beta;
        }
    }
}

