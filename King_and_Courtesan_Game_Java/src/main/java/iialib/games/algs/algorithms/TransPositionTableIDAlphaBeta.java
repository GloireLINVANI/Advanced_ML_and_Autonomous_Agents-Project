package iialib.games.algs.algorithms;

import iialib.games.algs.GameAlgorithm;
import iialib.games.algs.IHeuristic;
import iialib.games.model.IBoard;
import iialib.games.model.IMove;
import iialib.games.model.IRole;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class TransPositionTableIDAlphaBeta<Move extends IMove, Role extends IRole, Board extends IBoard<Move, Role, Board>> implements GameAlgorithm<Move, Role, Board> {
    // Constants
    /**
     * Default value for depth limit
     */
    private static final int DEPTH_MAX_DEFAUT = 6;

    // Attributes
    /**
     * Time limit for iterative deepening in milliseconds
     */
    private static final long DEFAULT_TIME_LIMIT = 30000;
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
    private final HashMap<AbstractMap.SimpleEntry<Integer, Integer>, AbstractMap.SimpleEntry<Integer, Integer>> transpositionTable = new HashMap<>(); // transposition table
    private final int playerMaxRoleHashCode; // hash code of the max player role
    private final int playerMinRoleHashCode; // hash code of the min player role
    /**
     * Algorithm max depth
     */
    private int depthMax = DEPTH_MAX_DEFAUT;
    private long timeLimit = DEFAULT_TIME_LIMIT; // time limit for iterative deepening in milliseconds
    private boolean timeOut; // true if the time limit is reached
    private long startTime; // start time of the algorithm thinking in milliseconds


    public TransPositionTableIDAlphaBeta(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h) {
        this.playerMaxRole = playerMaxRole;
        this.playerMinRole = playerMinRole;
        this.h = h;
        this.playerMaxRoleHashCode = this.playerMaxRole.hashCode();
        this.playerMinRoleHashCode = this.playerMinRole.hashCode();
    }

    public TransPositionTableIDAlphaBeta(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h, int depthMax) {
        this(playerMaxRole, playerMinRole, h);
        this.depthMax = depthMax;
    }

    public TransPositionTableIDAlphaBeta(Role playerMaxRole, Role playerMinRole, IHeuristic<Board, Role> h, int depthMax, long timeLimit) {
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
        // int[] heuristicValues = new int[newBoards.size()];
        // Arrays.fill(heuristicValues, Integer.MIN_VALUE);
        while (depth <= depthMax) {
            int currentDepthAlpha = Integer.MIN_VALUE;
            //  if (depth > 1) {
            //    newBoards.sort((b1, b2) -> Integer.compare(heuristicValues[newBoards.indexOf(b2)], heuristicValues[newBoards.indexOf(b1)]));
            // }
            orderMovesRoot(newBoards, playerMaxRole);
            for (AbstractMap.SimpleEntry<Board, Integer> boardIntegerSimpleEntry : newBoards) {
                Board newBoard = boardIntegerSimpleEntry.getKey();
                int moveValue = minMax(newBoard, depth, 1, currentDepthAlpha, beta);
                if (moveValue > currentDepthAlpha) {
                    currentDepthAlpha = moveValue;
                    bestMove = possibleMoves.get(boardIntegerSimpleEntry.getValue());
                    // heuristicValues[i] = moveValue;
                    AbstractMap.SimpleEntry<Integer, Integer> key = new AbstractMap.SimpleEntry<>(newBoard.hashCode(), playerMinRoleHashCode);
                    transpositionTable.put(key, new AbstractMap.SimpleEntry<>(depthMax - depth, moveValue));
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
        AbstractMap.SimpleEntry<Integer, Integer> key = new AbstractMap.SimpleEntry<>(board.hashCode(), playerMaxRoleHashCode);
        if (transpositionTable.containsKey(key) && transpositionTable.get(key).getKey() >= maxDepth - profondeur) {
            return transpositionTable.get(key).getValue();
        }

        if (board.isGameOver() || profondeur == maxDepth) {
            int value = this.h.eval(board, playerMaxRole);
            transpositionTable.put(key, new AbstractMap.SimpleEntry<>(maxDepth - profondeur, value));
            return value;
        }

        ArrayList<Move> possibleMoves = board.possibleMoves(playerMaxRole);
        ArrayList<Board> newBoards = new ArrayList<>();
        for (Move move : possibleMoves) {
            newBoards.add(board.play(move, playerMaxRole));
        }
        orderMoves(newBoards, playerMaxRole);
        for (Board newBoard : newBoards) {
            int moveValue = minMax(newBoard, maxDepth, profondeur + 1, alpha, beta);
            alpha = Math.max(alpha, moveValue);
            if (alpha >= beta) {
                return beta;
            }
        }
        transpositionTable.put(key, new AbstractMap.SimpleEntry<>(maxDepth - profondeur, alpha));
        return alpha;
    }

    private int minMax(Board board, int maxDepth, int profondeur, int alpha, int beta) {
        if (System.currentTimeMillis() - startTime >= timeLimit) {
            timeOut = true;
            return beta;
        }
        AbstractMap.SimpleEntry<Integer, Integer> key = new AbstractMap.SimpleEntry<>(board.hashCode(), playerMinRoleHashCode);
        if (transpositionTable.containsKey(key) && transpositionTable.get(key).getKey() >= maxDepth - profondeur) {
            return transpositionTable.get(key).getValue();
        }
        if (board.isGameOver() || profondeur == maxDepth) {
            int value = this.h.eval(board, playerMaxRole);
            transpositionTable.put(key, new AbstractMap.SimpleEntry<>(maxDepth - profondeur, value));
            return value;
        }
        ArrayList<Move> possibleMoves = board.possibleMoves(playerMinRole);
        ArrayList<Board> newBoards = new ArrayList<>();
        for (Move move : possibleMoves) {
            newBoards.add(board.play(move, playerMinRole));
        }
        orderMoves(newBoards, playerMinRole);
        for (Board newBoard : newBoards) {
            int moveValue = maxMin(newBoard, maxDepth, profondeur + 1, alpha, beta);
            beta = Math.min(beta, moveValue);
            if (beta <= alpha) {
                return alpha;
            }
        }
        transpositionTable.put(key, new AbstractMap.SimpleEntry<>(maxDepth - profondeur, beta));
        return beta;
    }

    private void orderMoves(List<Board> boards, Role playerRole) {
        boards.sort((b1, b2) -> {
            AbstractMap.SimpleEntry<Integer, Integer> key1 = new AbstractMap.SimpleEntry<>(b1.hashCode(), playerRole == playerMaxRole ? playerMinRoleHashCode : playerMaxRoleHashCode);
            AbstractMap.SimpleEntry<Integer, Integer> key2 = new AbstractMap.SimpleEntry<>(b2.hashCode(), playerRole == playerMaxRole ? playerMinRoleHashCode : playerMaxRoleHashCode);
            int v1 = transpositionTable.getOrDefault(key1, new AbstractMap.SimpleEntry<>(0, playerRole == playerMaxRole ? IHeuristic.MIN_VALUE : IHeuristic.MAX_VALUE)).getValue();
            int v2 = transpositionTable.getOrDefault(key2, new AbstractMap.SimpleEntry<>(0, playerRole == playerMaxRole ? IHeuristic.MIN_VALUE : IHeuristic.MAX_VALUE)).getValue();
            return playerRole == playerMaxRole ? Integer.compare(v2, v1) : Integer.compare(v1, v2);
        });
    }

    private void orderMovesRoot(List<AbstractMap.SimpleEntry<Board, Integer>> boards, Role playerRole) {
        boards.sort((b1, b2) -> {
            AbstractMap.SimpleEntry<Integer, Integer> key1 = new AbstractMap.SimpleEntry<>(b1.getKey().hashCode(), playerRole == playerMaxRole ? playerMinRoleHashCode : playerMaxRoleHashCode);
            AbstractMap.SimpleEntry<Integer, Integer> key2 = new AbstractMap.SimpleEntry<>(b2.getKey().hashCode(), playerRole == playerMaxRole ? playerMinRoleHashCode : playerMaxRoleHashCode);
            int v1 = transpositionTable.getOrDefault(key1, new AbstractMap.SimpleEntry<>(0, playerRole == playerMaxRole ? IHeuristic.MIN_VALUE : IHeuristic.MAX_VALUE)).getValue();
            int v2 = transpositionTable.getOrDefault(key2, new AbstractMap.SimpleEntry<>(0, playerRole == playerMaxRole ? IHeuristic.MIN_VALUE : IHeuristic.MAX_VALUE)).getValue();
            return playerRole == playerMaxRole ? Integer.compare(v2, v1) : Integer.compare(v1, v2);
        });
    }

}

