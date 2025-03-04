package iialib.games.algs.algorithms;

import iialib.games.algs.GameAlgorithm;
import iialib.games.algs.IHeuristic;
import iialib.games.model.IBoard;
import iialib.games.model.IMove;
import iialib.games.model.IRole;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;

public class IDAlphaBeta<Move extends IMove, Role extends IRole, Board extends IBoard<Move, Role, Board>> implements GameAlgorithm<Move, Role, Board> {
    // Constants
    /**
     * Default value for depth limit
     */
    private static final int DEPTH_MAX_DEFAUT = 8;

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
     * Time limit for iterative deepening in milliseconds
     */
    private static final long DEFAULT_TIME_LIMIT = 30000;
    /**
     * Heuristic used by the max player
     */
    private final IHeuristic<Board, Role> h;
    private long timeLimit = DEFAULT_TIME_LIMIT; // time limit for iterative deepening in milliseconds
    private boolean timeOut; // true if the time limit is reached
    private long startTime; // start time of the algorithm thinking in milliseconds


    public IDAlphaBeta(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h) {
        this.playerMaxRole = playerMaxRole;
        this.playerMinRole = playerMinRole;
        this.h = h;
    }

    public IDAlphaBeta(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h, int depthMax) {
        this(playerMaxRole, playerMinRole, h);
        this.depthMax = depthMax;
    }

    public IDAlphaBeta(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h, int depthMax, long timeLimit) {
        this(playerMaxRole, playerMinRole, h);
        this.depthMax = depthMax;
        this.timeLimit = timeLimit;
    }

    // --------- IAlgo METHODS ---------

    @Override
    public Move bestMove(Board board, Role playerRole) {
        startTime = System.currentTimeMillis();
        timeOut = false;
        Move bestMove = null;
        Move globalBestMove = null;
        int beta = Integer.MAX_VALUE;

        ArrayList<Move> possibleMoves = board.possibleMoves(playerMaxRole);
        ArrayList<AbstractMap.SimpleEntry<Board, Integer>> newBoards = new ArrayList<>();
        for (int i = 0; i < possibleMoves.size(); i++) {
            newBoards.add(new AbstractMap.SimpleEntry<>(board.play(possibleMoves.get(i), playerMaxRole), i));
        }
        int depth = 1;
        int[] heuristicValues = new int[newBoards.size()];
        Arrays.fill(heuristicValues, Integer.MIN_VALUE);
        while (depth <= depthMax) {
            int currentDepthAlpha = Integer.MIN_VALUE;
            if (depth > 1) {
                newBoards.sort((b1, b2) -> Integer.compare(heuristicValues[newBoards.indexOf(b2)], heuristicValues[newBoards.indexOf(b1)]));
            }
            for (int i = 0; i < newBoards.size(); i++) {
                int moveValue = minMax(newBoards.get(i).getKey(), depth, 1, currentDepthAlpha, beta);
                if (moveValue > currentDepthAlpha) {
                    currentDepthAlpha = moveValue;
                    bestMove = possibleMoves.get(newBoards.get(i).getValue());
                    heuristicValues[i] = moveValue;
                }
                if (timeOut) {
                    return globalBestMove;
                }
            }
            globalBestMove = bestMove;
            depth++;
        }
        return globalBestMove;
    }

    // --------- PRIVATE METHODS ---------
    private int maxMin(Board board, int maxDepth, int profondeur, int alpha, int beta) {
        if (System.currentTimeMillis() - startTime >= timeLimit) {
            timeOut = true;
            return alpha;
        }
        if (board.isGameOver() || profondeur == maxDepth) {
            return h.eval(board, playerMaxRole);
        } else {
            ArrayList<Move> coupsPossibles = board.possibleMoves(playerMaxRole);
            for (Move coupsPossible : coupsPossibles) {
                int moveValue = minMax(board.play(coupsPossible, playerMaxRole), maxDepth, profondeur + 1, alpha, beta);
                alpha = Math.max(alpha, moveValue);
                if (alpha >= beta) {
                    return beta;
                }
            }
            return alpha;
        }
    }

    private int minMax(Board board, int maxDepth, int profondeur, int alpha, int beta) {
        if (board.isGameOver() || profondeur == maxDepth) {
            return this.h.eval(board, playerMaxRole);
        } else {
            ArrayList<Move> coupsPossibles = board.possibleMoves(playerMinRole);
            for (Move coupsPossible : coupsPossibles) {
                int moveValue = maxMin(board.play(coupsPossible, playerMinRole), maxDepth, profondeur + 1, alpha, beta);
                beta = Math.min(beta, moveValue);
                if (beta <= alpha) {
                    return alpha;
                }
            }
            return beta;
        }
    }
}

