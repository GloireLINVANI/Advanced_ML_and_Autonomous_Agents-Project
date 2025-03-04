package games.kac;

import iialib.games.algs.IHeuristic;

import java.util.AbstractMap;
import java.util.ArrayList;

public class KingAndCourtesanHeuristics  {

    private static final int H_PIECES_DIFF = 40;
    private static final int H_POSSIBLE_MOVES_DIFF = 10;
    private static final int H_POTENTIAL_CAPTURE = 20;
    private static final int H_DIST_KING = 300;
    private static final int H_CLOSE_TO_VICTORY = 3000;
    private static final int H_WIN = Integer.MAX_VALUE;

    public static IHeuristic<KingAndCourtesanBoard, KingAndCourtesanRole> hRed = (board, role) -> eval(board, KingAndCourtesanRole.RED);

    public static IHeuristic<KingAndCourtesanBoard, KingAndCourtesanRole> hBlue = (board, role) -> eval(board, KingAndCourtesanRole.BLUE);

    public static int eval(KingAndCourtesanBoard board, KingAndCourtesanRole role) {
       // System.out.println(board);
        // Number of pieces of each color
        int piecesRed = 0;
        int piecesBlue = 0;

        // Number of possible moves for each color
        int coupsPossiblesRed = 0;
        int coupsPossiblesBlue = 0;

        // Number of potential captures for each color
        int potentielCaptureRed = 0;
        int potentielCaptureBlue = 0;

        // Distance between the king and the base of the opponent
        int distBlueKingBase = 0, distRedKingBase = 0;


        // Number of enemy pieces close to the king of each color
        int nbcloseToRedKing = 0, nbcloseToBlueKing = 0;

        // Wether our king is close to the king of the opponent
        boolean closeToRedKing, closeToBlueKing;

        // Wether our king is close to the base of the opponent
        boolean closeToRedKingBase;
        boolean closeToBlueKingBase;

        // Wether we're on a winning board or not
        boolean redWin = board.redWins();
        boolean blueWin = board.blueWins();

        if (!redWin && !blueWin) {
            KingAndCourtesanBoard.SQUARE[][] boardGrid = board.getBoardGrid();

            ArrayList<AbstractMap.SimpleEntry<Integer, Integer>> helper = new ArrayList<>();
            helper.add(new AbstractMap.SimpleEntry<>(-1, -1));
            helper.add(new AbstractMap.SimpleEntry<>(-1, 0));
            helper.add(new AbstractMap.SimpleEntry<>(-1, 1));
            helper.add(new AbstractMap.SimpleEntry<>(0, -1));
            helper.add(new AbstractMap.SimpleEntry<>(0, 1));
            helper.add(new AbstractMap.SimpleEntry<>(1, -1));
            helper.add(new AbstractMap.SimpleEntry<>(1, 0));
            helper.add(new AbstractMap.SimpleEntry<>(1, 1));

            double meanDistRedKing;
            double meanDistBlueKing;
            int sumDistRedKing = 0;
            int sumDistBlueKing = 0;
            int x;
            int y;
            int xRedKing = board.getRedKingPosition().x;
            int yRedKing = board.getRedKingPosition().y;
            int xBlueKing = board.getBlueKingPosition().x;
            int yBlueKing = board.getBlueKingPosition().y;

            for (int i = 0; i < boardGrid.length; i++) {
                for (int j = 0; j < boardGrid[i].length; j++) {
                    if (boardGrid[i][j] == KingAndCourtesanBoard.SQUARE.RED_KING || boardGrid[i][j] == KingAndCourtesanBoard.SQUARE.RED_COURTESAN) {
                        piecesRed++;
                        sumDistBlueKing += chebyshevDistance(i, j, xBlueKing, yBlueKing);
                        boolean isRedKing = xRedKing == i && yRedKing == j;
                        if (isRedKing) {
                            distBlueKingBase = chebyshevDistance(i, j, board.getBoardSize() - 1, board.getBoardSize() - 1);
                        }
                        for (AbstractMap.SimpleEntry<Integer, Integer> entry : helper) {
                            x = i + entry.getKey();
                            y = j + entry.getValue();
                            if (x >= 0 && x < boardGrid.length && y >= 0 && y < boardGrid[x].length) {
                                if (opponentPiece(KingAndCourtesanRole.RED, boardGrid[x][y])) {
                                    if (isRedKing) {
                                        nbcloseToRedKing++;
                                    }
                                    potentielCaptureRed++;
                                    coupsPossiblesRed++;
                                } else {
                                    KingAndCourtesanMove move = new KingAndCourtesanMove((char) ('A' + i), j, (char) ('A' + x), y);
                                    if (board.isValidMove(move, KingAndCourtesanRole.RED)) {
                                        coupsPossiblesRed++;
                                    }
                                }
                            }
                        }
                    } else if (boardGrid[i][j] == KingAndCourtesanBoard.SQUARE.BLUE_KING || boardGrid[i][j] == KingAndCourtesanBoard.SQUARE.BLUE_COURTESAN) {
                        piecesBlue++;
                        sumDistRedKing += chebyshevDistance(i, j, xRedKing, yRedKing);
                        boolean isBlueKing = xBlueKing == i && yBlueKing == j;
                        if (isBlueKing) {
                            distRedKingBase = chebyshevDistance(i, j, 0, 0);
                        }
                        for (AbstractMap.SimpleEntry<Integer, Integer> entry : helper) {
                            x = i + entry.getKey();
                            y = j + entry.getValue();
                            if (x >= 0 && x < boardGrid.length && y >= 0 && y < boardGrid[x].length) {
                                if (opponentPiece(KingAndCourtesanRole.BLUE, boardGrid[x][y])) {
                                    if (isBlueKing) {
                                        nbcloseToBlueKing++;
                                    }
                                    potentielCaptureBlue++;
                                    coupsPossiblesBlue++;
                                } else {
                                    KingAndCourtesanMove move = new KingAndCourtesanMove((char) ('A' + i), j, (char) ('A' + x), y);
                                    if (board.isValidMove(move, KingAndCourtesanRole.BLUE)) {
                                        coupsPossiblesBlue++;
                                    }
                                }
                            }
                        }
                    }
                }

            }
            meanDistRedKing = (double) sumDistRedKing / piecesBlue;
            if(meanDistRedKing < 1) {
                meanDistRedKing = 1;
            }
            meanDistBlueKing = (double) sumDistBlueKing / piecesRed;
            if(meanDistBlueKing < 1) {
                meanDistBlueKing = 1;
            }
            closeToRedKingBase = distRedKingBase == 1;
            closeToBlueKingBase = distBlueKingBase == 1;
            closeToRedKing = nbcloseToRedKing > 0;
            closeToBlueKing = nbcloseToBlueKing > 0;
            int diff_pieces;
            int diff_possible_moves;
            int diff_potential_capture;
            int king_base_progress;
            int opponent_king_capture_progress;
            int king_base_safety;
            int king_safety;
            if (role == KingAndCourtesanRole.RED) {
                diff_pieces = H_PIECES_DIFF * (piecesRed - piecesBlue);
                diff_possible_moves = H_POSSIBLE_MOVES_DIFF * (coupsPossiblesRed - coupsPossiblesBlue);
                diff_potential_capture = H_POTENTIAL_CAPTURE * (potentielCaptureRed - potentielCaptureBlue);
                king_base_progress = (closeToBlueKingBase ? H_CLOSE_TO_VICTORY : 0);
                opponent_king_capture_progress = (closeToBlueKing ? H_CLOSE_TO_VICTORY : 0);
                king_base_safety = (closeToRedKingBase ? -H_CLOSE_TO_VICTORY : 300);
                king_safety = (closeToRedKing ? -H_CLOSE_TO_VICTORY : 300);
                return diff_pieces + diff_possible_moves + diff_potential_capture - (H_DIST_KING /(int) meanDistRedKing) +
                        (H_DIST_KING /(int) meanDistBlueKing) - (H_DIST_KING /distRedKingBase) + (H_DIST_KING /distBlueKingBase) +
                        king_base_progress + opponent_king_capture_progress + king_base_safety + king_safety;
            } else {
                diff_pieces = H_PIECES_DIFF * (piecesBlue - piecesRed);
                diff_possible_moves = H_POSSIBLE_MOVES_DIFF * (coupsPossiblesBlue - coupsPossiblesRed);
                diff_potential_capture = H_POTENTIAL_CAPTURE * (potentielCaptureBlue - potentielCaptureRed);
                king_base_progress = (closeToRedKingBase ? H_CLOSE_TO_VICTORY : 0);
                opponent_king_capture_progress = (closeToRedKing ? H_CLOSE_TO_VICTORY : 0);
                king_base_safety = (closeToBlueKingBase ? -H_CLOSE_TO_VICTORY : 300);
                king_safety = (closeToBlueKing ? -H_CLOSE_TO_VICTORY : 300);
                return diff_pieces + diff_possible_moves + diff_potential_capture - (H_DIST_KING /(int) meanDistBlueKing) +
                         (H_DIST_KING /(int) meanDistRedKing) - (H_DIST_KING /distBlueKingBase) + (H_DIST_KING /distRedKingBase) +
                        king_base_progress + opponent_king_capture_progress + king_base_safety + king_safety;

            }
        }else {
            if(role == KingAndCourtesanRole.RED) {
                return redWin ? H_WIN : -H_WIN;
            } else {
                return blueWin ? H_WIN : -H_WIN;
            }
        }
    }

    public static int chebyshevDistance(int x1, int y1, int x2, int y2) {
        int dx = Math.abs(x2 - x1);
        int dy = Math.abs(y2 - y1);
        return Math.max(dx, dy);
    }


    private static boolean opponentPiece(KingAndCourtesanRole role, KingAndCourtesanBoard.SQUARE square) {
        if(role == KingAndCourtesanRole.RED) {
            return square == KingAndCourtesanBoard.SQUARE.BLUE_KING || square == KingAndCourtesanBoard.SQUARE.BLUE_COURTESAN;
        }else {
            return square == KingAndCourtesanBoard.SQUARE.RED_KING || square == KingAndCourtesanBoard.SQUARE.RED_COURTESAN;
        }
    }
}


