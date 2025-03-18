package iialib.games.contest;

import org.apache.commons.cli.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;

public class GameServer {

    public static final int CONNECTION_TIMEOUT = 15; // in seconds

    public static ServerSocket startServer(int portNumber) {
        try {
            ServerSocket serverSocket = new ServerSocket(portNumber);
            serverSocket.setSoTimeout(CONNECTION_TIMEOUT * 1000);
            return serverSocket;
        } catch (IOException e) {
            System.err.println("[ERROR] Server cannot read port " + portNumber);
            System.exit(2);
        }
        return null;
    }

    public static void run(String[] args, IRule game) {
        ArgParser parser = ArgParser.parse(args);
        System.out.println(parser.portNumber);
        ServerSocket serverSocket = startServer(parser.portNumber);
        try {
            Socket client_1 = serverSocket.accept();
            String client_1_ID = (new BufferedReader(new InputStreamReader(client_1.getInputStream()))).readLine();
            System.out.println("[REFEREE] Player 1 is " + client_1_ID);
            System.out.println("Waiting for 2nd client ...");

            Socket client_2 = serverSocket.accept();
            String client_2_ID = (new BufferedReader(new InputStreamReader(client_2.getInputStream()))).readLine();
            System.out.println("[REFEREE] Player 2 is " + client_2_ID);

            Thread gameThread = new Thread(new Referee(game, client_1, client_2));
            gameThread.start();

        } catch (SocketTimeoutException e) {
            System.err.println("[ERROR] SOCKET TIMEOUT : waited clients for more than " + CONNECTION_TIMEOUT + " sec.");
            System.exit(3);
        } catch (IOException e) {
            System.err.println("[ERROR] " + e);
            System.exit(4);
        }
        try {
            serverSocket.close();
        } catch (IOException e) {
            System.err.println("[ERROR] Cannot close server :" + e);
            System.exit(5);
        }
    }

    public static void run(String[] args, IRule game, AApplet gameView) {
        ArgParser parser = ArgParser.parse(args);
        System.out.println(parser.portNumber);
        ServerSocket serverSocket = startServer(parser.portNumber);
        try {
            Socket client_1 = serverSocket.accept();
            String client_1_ID = (new BufferedReader(new InputStreamReader(client_1.getInputStream()))).readLine();
            System.out.println("[REFEREE] Player 1 is " + client_1_ID);
            System.out.println("Waiting for 2nd client ...");

            Socket client_2 = serverSocket.accept();
            String client_2_ID = (new BufferedReader(new InputStreamReader(client_2.getInputStream()))).readLine();
            System.out.println("[REFEREE] Player 2 is " + client_2_ID);

            Thread gameThread = new Thread(new Referee(game, gameView, client_1, client_2, parser.useGraphicApp));
            gameThread.start();

        } catch (SocketTimeoutException e) {
            System.err.println("[ERROR] SOCKET TIMEOUT : waited clients for more than " + CONNECTION_TIMEOUT + " sec.");
            System.exit(3);
        } catch (IOException e) {
            System.err.println("[ERROR] " + e);
            System.exit(4);
        }
        try {
            serverSocket.close();
        } catch (IOException e) {
            System.err.println("[ERROR] Cannot close server :" + e);
            System.exit(5);
        }
    }

    static class ArgParser {
        static final String OPT_PORT_NUMBER = "port-number";
        static final String OPT_GRAPHIC_APP = "graphic-app";
        static final String OPT_QUIET = "quiet";
        final static Options options = buildOptions();

        int portNumber;
        boolean useGraphicApp = false;
        boolean quiet = false;

        private ArgParser(int portNumber, boolean useGraphicApp, boolean quiet) {
            this.portNumber = portNumber;
            this.useGraphicApp = useGraphicApp;
            this.quiet = quiet;
        }

        public static Options buildOptions() {
            Options options = new Options();
            options.addOption("help", "show this help message");
            options.addOption(Option.builder("p").longOpt(OPT_PORT_NUMBER).required().hasArg().desc("port number used to communicate").build());
            options.addOption(Option.builder("g").longOpt(OPT_GRAPHIC_APP).hasArg(false).desc("flag to activate graphic app").build());
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
            formatter.printHelp("runContest", options);
        }

        public static ArgParser parse(String[] args) {
            CommandLine line = parseArgs(args);

            boolean quiet = line.hasOption(OPT_QUIET);

            String portNumberStr = line.getOptionValue(OPT_PORT_NUMBER);
            int portNumber = Integer.parseInt(portNumberStr);
            if (!quiet)
                System.out.println("port number is " + portNumber);

            boolean use_graphic_app = line.hasOption(OPT_GRAPHIC_APP);
            if (!quiet)
                System.out.println("use_graphic_app is " + use_graphic_app);

            return new ArgParser(portNumber, use_graphic_app, quiet);
        }

    }

}
