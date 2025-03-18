package games.kac;

import iialib.games.algs.AIPlayer;
import iialib.games.algs.AbstractGame;

import java.util.ArrayList;

public class KingAndCourtesanGame extends AbstractGame<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard> {
    public KingAndCourtesanGame(ArrayList<AIPlayer<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard>> players, KingAndCourtesanBoard board) {
        super(players, board);
    }

    public static void main(String[] args) {

        MyChallenger challenger = new MyChallenger();
        challenger.setRole("RED");
        challenger.setBoardFromFile("example_file_to_read.txt");

        //KingAndCourtesanBoard board = new KingAndCourtesanBoard(9);
        System.out.println("Initial Board");
        System.out.println(challenger.getBoard());
        String best_move = challenger.bestMove();
        System.out.println("Best Move");
        System.out.println(best_move);
        challenger.iPlay(best_move);
        System.out.println("Board after my move");
        System.out.println(challenger.getBoard());
        challenger.saveBoardToFile("example_file_to_write.txt");

        //KingAndCourtesanBoard board = challenger.getBoard();
        //System.out.println(board);
        //System.out.println(board.isValidMove(new KingAndCourtesanMove("A0-B0"), KingAndCourtesanRole.RED));
        //System.out.println(board.isValidMove(new KingAndCourtesanMove("D0-C0"), KingAndCourtesanRole.RED));
        //System.out.println(board.possibleMoves(KingAndCourtesanRole.BLUE));
        //System.out.println(board.possibleMoves(KingAndCourtesanRole.RED));
        //board = board.play(new KingAndCourtesanMove("A0-B0"), KingAndCourtesanRole.RED);
        //System.out.println(board);
        // KingAndCourtesanBoard board1 = new KingAndCourtesanBoard(9);
        // KingAndCourtesanBoard board2 = new KingAndCourtesanBoard(board1);
    }
}
