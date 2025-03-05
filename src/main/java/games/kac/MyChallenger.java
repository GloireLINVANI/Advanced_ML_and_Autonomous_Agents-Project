package games.kac;

import iialib.games.algs.GameAlgorithm;
import iialib.games.algs.algorithms.IDAlphaBeta;
import iialib.games.model.IChallenger;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class MyChallenger implements IChallenger {
    private static final String teamName = "ID AlphaBeta";
    private KingAndCourtesanBoard board;
    private KingAndCourtesanRole role;
    private GameAlgorithm<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard> ai;

    @Override
    public String teamName() {
        return teamName;
    }

    @Override
    public void setRole(String role) {
        if (role.equals("RED")) {
            this.role = KingAndCourtesanRole.RED;
            this.ai = new IDAlphaBeta<>(this.role, KingAndCourtesanRole.BLUE, KingAndCourtesanHeuristics.hRed, 8);
           // this.ai = new TransPositionTableIDAlphaBeta<>(this.role, KingAndCourtesanRole.BLUE, KingAndCourtesanHeuristics.hRed, 6);
            //this.ai = new RandomChoice<>(this.role);
        } else {
            this.role = KingAndCourtesanRole.BLUE;
             this.ai = new IDAlphaBeta<>(this.role, KingAndCourtesanRole.RED, KingAndCourtesanHeuristics.hBlue, 8);
           // this.ai = new TransPositionTableIDAlphaBeta<>(this.role, KingAndCourtesanRole.RED, KingAndCourtesanHeuristics.hBlue, 6);
            //this.ai = new RandomChoice<>(this.role);
        }
        this.board = new KingAndCourtesanBoard();
    }

    @Override
    public void iPlay(String move) {
        KingAndCourtesanMove moveToPlay = new KingAndCourtesanMove(move);
        this.board = this.board.play(moveToPlay, this.role);

    }

    @Override
    public void otherPlay(String move) {
        KingAndCourtesanMove moveToPlay = new KingAndCourtesanMove(move);
        KingAndCourtesanRole otherRole;
        if (this.role == KingAndCourtesanRole.BLUE) {
            otherRole = KingAndCourtesanRole.RED;
        } else {
            otherRole = KingAndCourtesanRole.BLUE;
        }
        this.board = this.board.play(moveToPlay, otherRole);
    }

    @Override
    public String bestMove() {
        KingAndCourtesanMove bestMove = this.ai.bestMove(this.board, this.role);
        return bestMove.toString();
    }

    @Override
    public String victory() {
        return "L'équipe " + teamName() + " a gagné";
    }

    @Override
    public String defeat() {
        return "L'équipe " + teamName() + " a perdu";
    }

    @Override
    public String tie() {
        return "Tie";
    }

    @Override
    public String boardToString() {
        return this.board.toString();
    }

    /*public void saveBoardToFile(String fileName) {
        try {
            FileWriter writer = new FileWriter(fileName, false);
            writer.write(this.board.toString());
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }*/

    public void saveBoardToFile(String fileName) {
        try {
            FileWriter writer = new FileWriter(fileName, false);

            // Getting original board string
            String originalBoard = this.board.toString();
            String[] lines = originalBoard.split("\n");
            StringBuilder modifiedBoard = new StringBuilder();

            // Skipping the first line
            for (int i = 1; i < lines.length; i++) {
                String line = lines[i];

                // Checking if this is a board row (starts with a letter but not "BLUE" or "RED")
                if (!line.isEmpty() && Character.isLetter(line.charAt(0)) &&
                        !line.startsWith("BLUE") && !line.startsWith("RED")) {
                    // Removing the first character (row label)
                    modifiedBoard.append(line.substring(1)).append("\n");
                } else {
                    // Keeping king position information as is
                    modifiedBoard.append(line).append("\n");
                }
            }

            writer.write(modifiedBoard.toString().trim());
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void setBoardFromFile(String fileName) {
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(fileName));
            StringBuilder res = new StringBuilder();
            String line = reader.readLine();
            while (line != null) {
                res.append(line);
                res.append("\n");
                line = reader.readLine();
            }
            reader.close();
            this.board = new KingAndCourtesanBoard(res.toString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public Set<String> possibleMoves(String role) {
        ArrayList<KingAndCourtesanMove> moves;
        Set<String> res = new HashSet<>();
        if (role.equals("BLUE")) {
            moves = this.board.possibleMoves(KingAndCourtesanRole.BLUE);
        } else {
            moves = this.board.possibleMoves(KingAndCourtesanRole.RED);
        }
        for (KingAndCourtesanMove move : moves) {
            res.add(move.toString());
        }
        return res;
    }

    public KingAndCourtesanBoard getBoard() {
        return this.board;
    }
}