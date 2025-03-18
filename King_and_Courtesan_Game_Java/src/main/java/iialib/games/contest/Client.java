package iialib.games.contest;

import iialib.games.model.IChallenger;
import org.apache.commons.cli.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class Client {

    public static void main(String[] args) {
        ArgParser parser = ArgParser.parse(args);
        ChallengerSpeaker playerSpeacker = getSpeaker(parser.serverName, parser.portNumber);

        System.out.println("[CLIENT] Loading player class " + parser.className);
        IChallenger player = getChallenger(parser.className);

        System.out.println("[CLIENT] Send team name (" + player.teamName() + ") to server.");
        playerSpeacker.sendMessage(player.teamName());
        boolean isNotOver = true;
        try {
            while (isNotOver) {
                Message msg = new Message(playerSpeacker.getMessage());
                switch (msg.type) {
                    case ROLE:
                        player.setRole(msg.msg);
                        break;
                    case TEAM_NAME:
                        String teamName = player.teamName();
                        playerSpeacker.sendMessage(teamName);
                        break;
                    case I_PLAY:
                        player.iPlay(msg.msg);
                        break;
                    case OTHER_PLAY:
                        player.otherPlay(msg.msg);
                        break;
                    case BEST_MOVE:
                        String move = player.bestMove();
                        playerSpeacker.sendMessage(move);
                        break;
                    case VICTORY:
                        String victoryMessage = player.victory();
                        playerSpeacker.sendMessage(victoryMessage);
                        isNotOver = false;
                        break;
                    case DEFEAT:
                        String defeatMessage = player.defeat();
                        playerSpeacker.sendMessage(defeatMessage);
                        isNotOver = false;
                    case TIE:
                        String tieMessage = player.tie();
                        playerSpeacker.sendMessage(tieMessage);
                        isNotOver = false;
                }
            }

        } catch (IOException e) {
            System.err.println(e);
        }

    }


    static ChallengerSpeaker getSpeaker(String serverName, int portNumber) {
        try {
            return new ChallengerSpeaker(new Socket(serverName, portNumber));
        } catch (IOException e) {
            System.out.println("[ERROR] Cannot read " + serverName + " on port " + portNumber);
            System.out.println(e);
            System.exit(1);
        }
        return null;
    }

    static IChallenger getChallenger(String className) {
        Class<?> playerClass = null;
        try {
            playerClass = Class.forName(className);
        } catch (ClassNotFoundException e) {
            System.err.println(e);
            System.exit(2);
        }
        try {
            return (IChallenger) playerClass.newInstance();
        } catch (InstantiationException | IllegalAccessException e) {
            System.err.println("[ERROR] cannot instanciate " + playerClass);
            System.err.println(e);
            System.exit(3);
        }
        return null;
    }


    static class ArgParser {
        static final String OPT_PORT_NUMBER = "port-number";
        static final String OPT_PLAYER_CLASS = "class";
        static final String OPT_SERVER_NAME = "server-name";
        static final String OPT_QUIET = "quiet";
        final static Options options = buildOptions();

        int portNumber;
        String className;
        String serverName;
        boolean quiet = false;

        private ArgParser(int portNumber, String className, String serverName, boolean quiet) {
            this.portNumber = portNumber;
            this.className = className;
            this.serverName = serverName;
            this.quiet = quiet;
        }

        public static Options buildOptions() {
            Options options = new Options();
            options.addOption("help", "show this help message");
            options.addOption(Option.builder("p").longOpt(OPT_PORT_NUMBER).required().hasArg().desc("port number used to communicate").build());
            options.addOption(Option.builder("c").longOpt(OPT_PLAYER_CLASS).required().hasArg().desc("class of the player").build());
            options.addOption(Option.builder("s").longOpt(OPT_SERVER_NAME).required().hasArg().desc("name of the server ('localhost' probably)").build());
            options.addOption(Option.builder("q").longOpt(OPT_QUIET).hasArg(false).desc("flag making execution non verbose").build());

            return options;
        }

        public static CommandLine parseArgs(String[] args) {
            // create the parser
            CommandLineParser parser = new DefaultParser();
            try {
                // parse the command line arguments
                return parser.parse(options, args);
            } catch (ParseException e) {
                System.err.println("Parsing failed.  Reason: " + e.getMessage());
                printHelp();
                System.exit(1);
            }
            return null;
        }

        public static void printHelp() {
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("runClient", options);
        }

        public static ArgParser parse(String[] args) {
            CommandLine line = parseArgs(args);

            boolean quiet = line.hasOption(OPT_QUIET);

            String portNumberStr = line.getOptionValue(OPT_PORT_NUMBER);
            int portNumber = Integer.parseInt(portNumberStr);
            if (!quiet)
                System.out.println("port number is " + portNumber);

            String className = line.getOptionValue(OPT_PLAYER_CLASS);
            if (!quiet) {
                System.out.println("className is " + className);
            }

            String serverName = line.getOptionValue(OPT_SERVER_NAME);
            if (!quiet) {
                System.out.println("serverName is " + serverName);
            }

            return new Client.ArgParser(portNumber, className, serverName, quiet);
        }

    }

}

class ChallengerSpeaker {
    public Socket socket;
    public PrintWriter out;
    public BufferedReader in;
    String role;
    long time;

    public ChallengerSpeaker(Socket socket) throws IOException {
        this.socket = socket;
        out = new PrintWriter(socket.getOutputStream(), true);
        in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
    }

    String getMessage() throws IOException {
        return in.readLine();
    }

    void sendMessage(String msg) {
        out.println(msg);
    }


}
