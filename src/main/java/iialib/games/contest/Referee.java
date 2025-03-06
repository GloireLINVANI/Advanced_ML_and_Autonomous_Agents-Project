package iialib.games.contest;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.util.Date;

public class Referee implements Runnable {
    // applet game viewer (if required)
    static private AApplet gameView;
    static private JFrame f = null;
    IRule game;
    ChallengerListener player_1;
    ChallengerListener player_2;
    boolean UseGraphicApp;

    public Referee(IRule game, Socket client1, Socket client2) throws IOException {
        this.game = game;
        client1.setSoTimeout(game.getTimeout() * 1000);
        client2.setSoTimeout(game.getTimeout() * 1000);
        this.player_1 = new ChallengerListener(client1, game.getFirstRole());
        this.player_2 = new ChallengerListener(client2, game.getSecondRole());
        this.UseGraphicApp = false;
    }

    public Referee(IRule game, AApplet gameView, Socket client1, Socket client2, boolean UseGraphicApp) throws IOException {
        this.game = game;
        Referee.gameView = gameView;
        client1.setSoTimeout(game.getTimeout() * 1000);
        client2.setSoTimeout(game.getTimeout() * 1000);
        this.player_1 = new ChallengerListener(client1, game.getFirstRole());
        this.player_2 = new ChallengerListener(client2, game.getSecondRole());
        this.UseGraphicApp = UseGraphicApp;

        // S'il le faut, on initialise l'applet graphique
        if (this.UseGraphicApp) {
            f = new JFrame("Vue du jeu");
            gameView.buildUI(f.getContentPane());
            f.setSize(gameView.getDimension());
            gameView.setMyFrame(f);
            f.setVisible(true);
            gameView.addBoard("Départ ", this.game.getBoard());
            gameView.update(f.getGraphics(), f.getInsets());
        }

    }

    @Override
    public void run() {
        player_1.sendRole();
        player_2.sendRole();

        player_1.setTime(game.getTotalTimeout() * 1000L);
        player_2.setTime(game.getTotalTimeout() * 1000L);
        ChallengerListener currentPlayer = player_1;
        ChallengerListener nextPlayer = player_2;
        GameOver gameOver;
        try {
            System.out.println("[REFEREE] " + player_1.teamName() + " is playing as " + player_1.getRole());
            System.out.println("[REFEREE] " + player_2.teamName() + " is playing as " + player_2.getRole());
            System.out.println("[REFEREE] C'est l'heure du dududuDUDU DU DUEL !!");
            int i = 0;
            String move = "A0-A0";
            System.out.println("\n");
            System.out.println("Starting board -");
            System.out.println("\n");
            while (true) {
                i++;
                System.out.println(game);
                System.out.println("==== Move " + i + " ====");

                final long start = new Date().getTime();
                try {
                    move = currentPlayer.bestMove();
                } catch (SocketTimeoutException e) {
                    gameOver = GameOver.TIMEOUT;
                    break;
                }
                final long stop = new Date().getTime();
                final long elapsedTime = stop - start;
                currentPlayer.reduceTime(elapsedTime);
                System.out.println("[REFEREE] " + currentPlayer.getRole() + " Took " + elapsedTime + " milliseconds to think");
                if (currentPlayer.getTime() <= 0) {
                    gameOver = GameOver.TOTAL_TIMEOUT;
                    break;
                }
                if (move == null) {
                    System.out.println("[REFEREE] " + currentPlayer.getRole() + " can't move anymore");
                    gameOver = GameOver.DEFEAT;
                    break;
                }
                if (!game.isValidMove(move, currentPlayer.getRole())) {
                    System.out.println("[REFEREE] " + currentPlayer.getRole() + " is trying an illegal move : " + move);
                    gameOver = GameOver.ILLEGAL_MOVE;
                    break;
                } else {
                    System.out.println("[REFEREE] " + currentPlayer.getRole() + " plays : " + move);
                    game = game.play(move, currentPlayer.getRole());
                    // Update the applet
                    if (UseGraphicApp) {
                        gameView.addBoard(move, game.getBoard());
                        gameView.update(f.getGraphics(), f.getInsets());
                    }
                }
                if (game.isGameOver()) {
                    if (game.isTie())
                        gameOver = GameOver.TIE;
                    else
                        gameOver = GameOver.VICTORY;
                    break;
                } else {
                    currentPlayer.play(move, currentPlayer.getRole());
                    nextPlayer.play(move, currentPlayer.getRole());

                    // SWAP PLAYER
                    ChallengerListener tmpSwap = currentPlayer;
                    currentPlayer = nextPlayer;
                    nextPlayer = tmpSwap;
                }
            }
            System.out.println(game);
            System.out.println("\n");
            System.out.println("==== END ====");
            handleGameOver(gameOver, currentPlayer, nextPlayer);
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    void handleGameOver(GameOver gameOver, ChallengerListener currentPlayer, ChallengerListener nextPlayer) throws IOException {
        if (gameOver == null) {
            System.out.println("[ERROR] Game crashed !");
            return;
        }
        switch (gameOver) {
            case TIMEOUT:
                System.out.println("[REFEREE] Timeout for player " + currentPlayer.getRole());
                System.out.println("[PLAYER] " + currentPlayer.teamName() + " : " + currentPlayer.defeat());
                // System.out.println("[PLAYER] "+currentPlayer.teamName()+" : "+ currentPlayer.defeat() + " " + currentPlayer.timeout());
                System.out.println("[PLAYER] " + nextPlayer.teamName() + " : " + nextPlayer.victory());
                // System.out.println("[PLAYER] "+nextPlayer.teamName()+" : " + nextPlayer.victory() + " " + nextPlayer.timeout());
                if (nextPlayer == player_1) {
                    System.out.println("VICTORY-TIMEOUT DEFEAT-TIMEOUT " + player_1.getTime() + " " + player_2.getTime());
                } else {
                    System.out.println("DEFEAT-TIMEOUT VICTORY-TIMEOUT " + player_1.getTime() + " " + player_2.getTime());
                }
                break;
            case TOTAL_TIMEOUT:
                System.out.println("[REFEREE] Total timeout for player " + currentPlayer.getRole());
                System.out.println("[PLAYER] " + currentPlayer.teamName() + " : " + currentPlayer.defeat());
                // System.out.println("[PLAYER] "+currentPlayer.teamName()+" : "+ currentPlayer.defeat() + " " + currentPlayer.timeout());
                System.out.println("[PLAYER] " + nextPlayer.teamName() + " : " + nextPlayer.victory());
                // System.out.println("[PLAYER] "+nextPlayer.teamName()+" : " + nextPlayer.victory() + " " + nextPlayer.timeout());
                if (nextPlayer == player_1) {
                    System.out.println("VICTORY-TOTALTIMEOUT DEFEAT-TOTALTIMEOUT " + player_1.getTime() + " " + player_2.getTime());
                } else {
                    System.out.println("DEFEAT-TOTALTIMEOUT VICTORY-TOTALTIMEOUT " + player_1.getTime() + " " + player_2.getTime());
                }
                break;
            case ILLEGAL_MOVE:
                System.out.println("[REFEREE] Illegal move from player " + currentPlayer.getRole());
                System.out.println("[PLAYER] " + currentPlayer.teamName() + " : " + currentPlayer.defeat());
                // System.out.println("[PLAYER] "+currentPlayer.teamName()+" : " + currentPlayer.defeat() + " " + currentPlayer.illegal_move());
                System.out.println("[PLAYER] " + nextPlayer.teamName() + " : " + nextPlayer.victory());
                // System.out.println("[PLAYER] "+nextPlayer.teamName()+" : " + nextPlayer.victory() + " " + nextPlayer.illegal_move());
                if (nextPlayer == player_1) {
                    System.out.println("VICTORY-ILLEGALMOVE DEFEAT-ILLEGALMOVE " + player_1.getTime() + " " + player_2.getTime());
                } else {
                    System.out.println("DEFEAT-ILLEGALMOVE VICTORY-ILLEGALMOVE " + player_1.getTime() + " " + player_2.getTime());
                }
                break;
            case DEFEAT:
                System.out.println("[REFEREE] It's a defeat !");
                System.out.println("[PLAYER] " + currentPlayer.teamName() + " : " + currentPlayer.defeat());
                System.out.println("[PLAYER] " + nextPlayer.teamName() + " : " + nextPlayer.victory());
                if (nextPlayer == player_1) {
                    System.out.println("VICTORY DEFEAT " + player_1.getTime() + " " + player_2.getTime());
                } else {
                    System.out.println("DEFEAT VICTORY " + player_1.getTime() + " " + player_2.getTime());
                }
                break;
            case TIE:
                System.out.println("[REFEREE] It's a tie !");
                System.out.println("[PLAYER] " + currentPlayer.teamName() + " : " + currentPlayer.tie());
                System.out.println("[PLAYER] " + nextPlayer.teamName() + " : " + nextPlayer.tie());
                System.out.println("TIE TIE " + player_1.getTime() + " " + player_2.getTime());
                break;
            case VICTORY:
                System.out.println("[REFEREE] the game is over ? " + game.isGameOver());
                if (game.isWinner(player_1.getRole())) {
                    System.out.println("[REFEREE] " + player_1.teamName() + " is the winner !");
                    System.out.println("[PLAYER] " + player_1.teamName() + " : " + player_1.victory());
                    System.out.println("[PLAYER] " + player_2.teamName() + " : " + player_2.defeat());
                    System.out.println("VICTORY DEFEAT " + player_1.getTime() + " " + player_2.getTime());
                } else {
                    System.out.println("[REFEREE] " + player_2.teamName() + " is the winner !");
                    System.out.println("[PLAYER] " + player_2.teamName() + " : " + player_2.victory());
                    System.out.println("[PLAYER] " + player_1.teamName() + " : " + player_1.defeat());
                    System.out.println("DEFEAT VICTORY " + player_1.getTime() + " " + player_2.getTime());
                }
                break;
        }

    }

    enum GameOver {TIMEOUT, TOTAL_TIMEOUT, ILLEGAL_MOVE, VICTORY, DEFEAT, TIE}

}

class ChallengerListener {
    public Socket socket;
    public PrintWriter out;
    public BufferedReader in;
    String role;
    long time;
    String name;

    public ChallengerListener(Socket socket, String role) throws IOException {
        this.socket = socket;
        out = new PrintWriter(socket.getOutputStream(), true);
        in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        this.role = role;

        sendMessage(Message.teamName());
        name = getMessage();
    }

    String getMessage() throws IOException {
        return in.readLine();
    }

    void sendMessage(String msg) {
        out.println(msg);
    }

    public String getRole() {
        return role;
    }

    void sendRole() {
        sendMessage(Message.role(role));
    }

    String bestMove() throws IOException {
        sendMessage(Message.bestMove());
        return getMessage();
    }

    void play(String move, String role) {
        if (role.equals(this.role))
            sendMessage(Message.iPlay(move));
        else
            sendMessage(Message.otherPlay(move));
    }

    String victory() {
        try {
            sendMessage(Message.victory());
            return getMessage();
        } catch (IOException e) {
            return "V";
        }
    }

    String defeat() {
        try {
            sendMessage(Message.defeat());
            return getMessage();
        } catch (IOException e) {
            return "D";
        }
    }

    // String timeout() {
    // 	try{
    // 		sendMessage(Message.timeout());
    // 		return getMessage();
    // 	}catch (IOException e){
    //         return "Timeout";
    //     }
    // }

    // String illegal_move() {
    // 	try{
    // 		sendMessage(Message.illegal_move());
    // 		return getMessage();
    // 	}catch (IOException e){
    //         return "Illegal_move";
    //     }
    // }

    public long getTime() {
        return time;
    }

    public void setTime(long time) {
        this.time = time;
    }

    public void reduceTime(long timeLost) {
        time -= timeLost;
    }

    public String tie() {
        try {
            sendMessage(Message.tie());
            return getMessage();
        } catch (IOException e) {
            return "T";
        }
    }

    public String teamName() {
        return name;
    }
}
