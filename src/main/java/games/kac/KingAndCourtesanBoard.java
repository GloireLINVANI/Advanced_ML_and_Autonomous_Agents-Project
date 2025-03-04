package games.kac;

import iialib.games.model.IBoard;
import iialib.games.model.Score;

import java.awt.*;
import java.util.AbstractMap;
import java.util.ArrayList;

public class KingAndCourtesanBoard implements IBoard<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard> {
    private final int boardSize;

    private final SQUARE[][] boardGrid;
    private Point redKingPosition;
    private Point blueKingPosition;
    private boolean hasBlueKing;
    private boolean hasRedKing;

    public KingAndCourtesanBoard(int boardSize) {
        this.boardSize = boardSize;
        this.boardGrid = new SQUARE[this.boardSize][this.boardSize];
        for (int i = 0; i < this.boardSize; i++) {
            for (int j = 0; j < this.boardSize; j++) {
                this.boardGrid[i][j] = SQUARE.EMPTY;
                if (j < this.boardSize - 1 - i) {
                    this.boardGrid[i][j] = SQUARE.RED_COURTESAN;
                } else if (j > this.boardSize - 1 - i) {
                    this.boardGrid[i][j] = SQUARE.BLUE_COURTESAN;
                }
            }
        }
        this.boardGrid[this.boardSize - 1][this.boardSize - 1] = SQUARE.BLUE_KING;
        this.boardGrid[0][0] = SQUARE.RED_KING;

        this.redKingPosition = new Point(0, 0);
        this.blueKingPosition = new Point(this.boardSize - 1, this.boardSize - 1);

        this.hasBlueKing = true;
        this.hasRedKing = true;
    }

    public KingAndCourtesanBoard() {
        this(6);
    }

    public KingAndCourtesanBoard(KingAndCourtesanBoard other) {
        this.boardSize = other.boardSize;
        this.redKingPosition = new Point(other.redKingPosition.x, other.redKingPosition.y);
        this.blueKingPosition = new Point(other.blueKingPosition.x, other.blueKingPosition.y);
        this.boardGrid = new SQUARE[this.boardSize][this.boardSize];
        for (int i = 0; i < this.boardSize; i++) {
            System.arraycopy(other.boardGrid[i], 0, this.boardGrid[i], 0, this.boardSize);
        }

        this.hasBlueKing = other.hasBlueKing;
        this.hasRedKing = other.hasRedKing;
    }

    public KingAndCourtesanBoard(String board) {
        this.hasBlueKing = false;
        this.hasRedKing = false;
        String[] rows = board.split("\n");
        this.boardSize = rows[0].length();
        this.boardGrid = new SQUARE[this.boardSize][this.boardSize];
        for (int i = 0; i < this.boardSize; i++) {
            for (int j = 0; j < this.boardSize; j++) {
                this.boardGrid[this.boardSize - 1 - i][j] = SQUARE.EMPTY;
                if (rows[i].charAt(j) == 'B') {
                    this.boardGrid[this.boardSize - 1 - i][j] = SQUARE.BLUE_COURTESAN;
                } else if (rows[i].charAt(j) == 'R') {
                    this.boardGrid[this.boardSize - 1 - i][j] = SQUARE.RED_COURTESAN;
                }
            }
        }
        if (rows.length >= this.boardSize + 2) {
            int helper, x, y;
            helper = rows[this.boardSize].length() - 1;
            x = Character.getNumericValue(rows[this.boardSize].charAt(helper - 3));
            y = Character.getNumericValue(rows[this.boardSize].charAt(helper - 1));
            this.boardGrid[x][y] = SQUARE.BLUE_KING;
            this.blueKingPosition = new Point(x, y);
            this.hasBlueKing = true;

            helper = rows[this.boardSize + 1].length() - 1;
            x = Character.getNumericValue(rows[this.boardSize + 1].charAt(helper - 3));
            y = Character.getNumericValue(rows[this.boardSize + 1].charAt(helper - 1));
            this.boardGrid[x][y] = SQUARE.RED_KING;
            this.redKingPosition = new Point(x, y);
            this.hasRedKing = true;
        }
    }

    // --------------------- IBoard Methods ---------------------

    public static boolean belongsToPlayer(SQUARE square, KingAndCourtesanRole playerRole) {
        if (playerRole == KingAndCourtesanRole.RED) {
            return square == SQUARE.RED_KING || square == SQUARE.RED_COURTESAN;
        } else if (playerRole == KingAndCourtesanRole.BLUE) {
            return square == SQUARE.BLUE_KING || square == SQUARE.BLUE_COURTESAN;
        }
        return false;
    }

    private static boolean squaresBelongToDifferentPlayers(SQUARE from, SQUARE to) {
        boolean fromIsRed = from == SQUARE.RED_KING || from == SQUARE.RED_COURTESAN;
        boolean toIsRed = to == SQUARE.RED_KING || to == SQUARE.RED_COURTESAN;
        return fromIsRed != toIsRed;
    }

    @Override
    public boolean isValidMove(KingAndCourtesanMove move, KingAndCourtesanRole playerRole) {
        int fromLig = move.getFromLig() - 'A';
        int fromCol = move.getFromCol();
        int toLig = move.getToLig() - 'A';
        int toCol = move.getToCol();

        SQUARE from = this.boardGrid[fromLig][fromCol];
        SQUARE to = this.boardGrid[toLig][toCol];

        int helper = fromLig - toLig;
        int helper2 = fromCol - toCol;

        // DÉPLACEMENT
        if (to == SQUARE.EMPTY) {
            if (from == SQUARE.RED_COURTESAN || from == SQUARE.RED_KING) {
                if (helper == 0) {
                    return helper2 == -1;
                }
                if (helper == -1) {
                    return helper2 == 0 || helper2 == -1;
                }
            } else {
                if (helper == 0) {
                    return helper2 == 1;
                }
                if (helper == 1) {
                    return helper2 == 0 || helper2 == 1;
                }
            }
        }

        // CAPTURE
        else if (squaresBelongToDifferentPlayers(from, to)) {
            return true;

            // ÉCHANGE
            } else {
            if (from == SQUARE.RED_KING) {
                    if (helper == 0) {
                        return helper2 == -1;
                    }
                    if (helper == -1) {
                        return helper2 == -1 || helper2 == 0;
                    }
            } else if (from == SQUARE.BLUE_KING) {
                    if (helper == 0) {
                        return helper2 == 1;
                    }
                    if (helper == 1) {
                        return helper2 == 0 || helper2 == 1;
                    }
                }
        }
        return false;

    }

    @Override
    public ArrayList<KingAndCourtesanMove> possibleMoves(KingAndCourtesanRole playerRole) {
        ArrayList<KingAndCourtesanMove> res = new ArrayList<>();
        ArrayList<AbstractMap.SimpleEntry<Integer, Integer>> helper = new ArrayList<>();
        helper.add(new AbstractMap.SimpleEntry<>(-1, -1));
        helper.add(new AbstractMap.SimpleEntry<>(-1, 0));
        helper.add(new AbstractMap.SimpleEntry<>(-1, 1));
        helper.add(new AbstractMap.SimpleEntry<>(0, -1));
        helper.add(new AbstractMap.SimpleEntry<>(0, 1));
        helper.add(new AbstractMap.SimpleEntry<>(1, -1));
        helper.add(new AbstractMap.SimpleEntry<>(1, 0));
        helper.add(new AbstractMap.SimpleEntry<>(1, 1));

        int x, y;
        char fromLig;
        char toLig;
        KingAndCourtesanMove move;
        for (int i = 0; i < this.boardSize; i++) {
            for (int j = 0; j < this.boardSize; j++) {
                if (this.boardGrid[i][j] != SQUARE.EMPTY && belongsToPlayer(this.boardGrid[i][j], playerRole)) {
                    for (AbstractMap.SimpleEntry<Integer, Integer> pair : helper) {
                        x = i + pair.getKey();
                        y = j + pair.getValue();
                        if (this.isValidCoord(x, y)) {
                            fromLig = (char) ('A' + i);
                            toLig = (char) ('A' + x);
                            move = new KingAndCourtesanMove(fromLig, j, toLig, y);
                            if (this.isValidMove(move, playerRole)) {
                                res.add(move);
                            }
                        }
                    }
                }
            }
        }
        return res;
    }

    @Override
    public boolean isGameOver() {
        return redWins() || blueWins();
    }

    @Override
    public ArrayList<Score<KingAndCourtesanRole>> getScores() {
        ArrayList<Score<KingAndCourtesanRole>> res = new ArrayList<>();
        if (this.isGameOver()) {
            if (redWins()) {
                res.add(new Score<>(KingAndCourtesanRole.RED, Score.Status.WIN, 1));
                res.add(new Score<>(KingAndCourtesanRole.BLUE, Score.Status.LOOSE, 0));
            } else {
                res.add(new Score<>(KingAndCourtesanRole.RED, Score.Status.LOOSE, 0));
                res.add(new Score<>(KingAndCourtesanRole.BLUE, Score.Status.WIN, 1));

            }
        }
        return res;
    }

    private boolean isValidCoord(int lig, int col) {
        return lig >= 0 && lig < this.boardSize && col >= 0 && col < this.boardSize;
    }

    /*Play a move, doesn't have to check if that move is valid
     * because we only play moves returned by possibleMoves method
     */
    @Override
    public KingAndCourtesanBoard play(KingAndCourtesanMove move, KingAndCourtesanRole playerRole) {
        KingAndCourtesanBoard res = new KingAndCourtesanBoard(this);
        int fromLig = move.getFromLig() - 'A';
        int fromCol = move.getFromCol();
        int toLig = move.getToLig() - 'A';
        int toCol = move.getToCol();

        SQUARE from = this.boardGrid[fromLig][fromCol];
        SQUARE to = this.boardGrid[toLig][toCol];

        if (to != SQUARE.EMPTY) {

            //CAPTURE
            if (squaresBelongToDifferentPlayers(from, to)) {
                if (from == SQUARE.BLUE_KING || from == SQUARE.BLUE_COURTESAN) {
                    if (to == SQUARE.RED_KING) {
                        res.hasRedKing = false;
                        res.redKingPosition = null;
                    }
                } else {
                    if (to == SQUARE.BLUE_KING) {
                        res.hasBlueKing = false;
                        res.blueKingPosition = null;
                    }
                }
                res.boardGrid[toLig][toCol] = from;
                res.boardGrid[fromLig][fromCol] = SQUARE.EMPTY;

                if(from == SQUARE.RED_KING) {
                    res.redKingPosition = new Point(toLig, toCol);
                }else if(from == SQUARE.BLUE_KING) {
                    res.blueKingPosition = new Point(toLig, toCol);
                }


                //ÉCHANGE
            } else {
                SQUARE temp = res.boardGrid[toLig][toCol];
                res.boardGrid[toLig][toCol] = from;
                res.boardGrid[fromLig][fromCol] = temp;
                if (playerRole == KingAndCourtesanRole.RED) {
                    res.redKingPosition = new Point(toLig, toCol);
                } else {
                    res.blueKingPosition = new Point(toLig, toCol);
                }
            }

            //DÉPLACEMENT
        } else {
            res.boardGrid[toLig][toCol] = from;
            res.boardGrid[fromLig][fromCol] = SQUARE.EMPTY;
            if (from == SQUARE.RED_KING){
                res.redKingPosition = new Point(toLig, toCol);
            } else if (from == SQUARE.BLUE_KING){
                res.blueKingPosition = new Point(toLig, toCol);
            }
        }
        return res;
    }

    // --------------------- Other Methods ---------------------

    public String toString() {
        StringBuilder res = new StringBuilder();
        res.append(" 012345\n");
        for (int i = this.boardSize - 1; i >= 0; i--) {
            res.append((char) ('A' + i));
            for (int j = 0; j < this.boardSize; j++) {
                if (boardGrid[i][j] == SQUARE.BLUE_COURTESAN || boardGrid[i][j] == SQUARE.BLUE_KING) {
                    res.append("B");
                } else if (boardGrid[i][j] == SQUARE.RED_COURTESAN || boardGrid[i][j] == SQUARE.RED_KING) {
                    res.append("R");
                } else {
                    res.append("-");
                }
            }
            res.append("\n");
        }
        if (blueKingPosition != null) {
            res.append("BLUE KING Position: (").append(blueKingPosition.x).append(",").append(blueKingPosition.y).append(")\n");
        }
        if (redKingPosition != null) {
            res.append("RED KING Position: (").append(redKingPosition.x).append(",").append(redKingPosition.y).append(")");
        }
        return res.toString();
    }

    public boolean redWins() {
        return (this.hasRedKing && !this.hasBlueKing) ||
                (this.boardGrid[this.boardSize - 1][this.boardSize - 1] == SQUARE.RED_KING);
    }

    public boolean blueWins() {
        return (this.hasBlueKing && !this.hasRedKing) ||
                (this.boardGrid[0][0] == SQUARE.BLUE_KING);
    }

    public enum SQUARE {
        RED_KING, BLUE_KING, BLUE_COURTESAN, RED_COURTESAN, EMPTY
    }

    public SQUARE[][] getBoardGrid() {
        return boardGrid;
    }

    public int getBoardSize() {
        return boardSize;
    }

    public Point getRedKingPosition() {
        return redKingPosition;
    }

    public Point getBlueKingPosition() {
        return blueKingPosition;
    }

   /* @Override
    public int hashCode() {
        int result = 1;
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                result = 42 * result + boardGrid[i][j].hashCode();
            }
        }
        return result;
    }*/
}
