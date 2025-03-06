package games.kac;

import iialib.games.algs.GameAlgorithm;
import iialib.games.algs.algorithms.IDAlphaBeta;
import org.apache.commons.cli.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;

/**
 * Server that handles ID Alpha-Beta algorithm calculations for King and Courtesan game.
 * This server is optimized for performance with dedicated algorithm instances per client.
 */
public class IDAlphaBetaServer {
    private static final int DEFAULT_PORT = 43;
    private static int searchDepth = 8;
    private static long timeLimit = 10000; // milliseconds
    private static boolean useTranspositionTable = true;

    public static void main(String[] args) {
        // Parse command line arguments
        CommandLine cmd = parseCommandLineArgs(args);

        // Get port number from arguments or use default
        int port = DEFAULT_PORT;
        if (cmd.hasOption('p')) {
            try {
                port = Integer.parseInt(cmd.getOptionValue('p'));
            } catch (NumberFormatException e) {
                System.err.println("Invalid port number, using default port " + DEFAULT_PORT);
            }
        }

        // Get search depth if specified
        if (cmd.hasOption('d')) {
            try {
                searchDepth = Integer.parseInt(cmd.getOptionValue('d'));
                System.out.println("Using search depth: " + searchDepth);
            } catch (NumberFormatException e) {
                System.err.println("Invalid search depth, using default: " + searchDepth);
            }
        }

        // Get time limit if specified
        if (cmd.hasOption('t')) {
            try {
                timeLimit = Long.parseLong(cmd.getOptionValue('t'));
                System.out.println("Using time limit: " + timeLimit + "ms");
            } catch (NumberFormatException e) {
                System.err.println("Invalid time limit, using default: " + timeLimit + "ms");
            }
        }


        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("ID Alpha-Beta server started on port " + port);
            System.out.println("Configuration: depth=" + searchDepth + ", timeLimit=" + timeLimit + "ms, transpositionTable=" + useTranspositionTable);

            // Main server loop - accept and handle client connections
            while (true) {
                try {
                    Socket clientSocket = serverSocket.accept();
                    System.out.println("Client connected to Alpha-Beta server from " + clientSocket.getInetAddress());

                    // Create a new thread to handle this client
                    Thread clientThread = new Thread(new ClientHandler(clientSocket));
                    clientThread.start();

                } catch (Exception e) {
                    System.err.println("Error accepting client connection: " + e.getMessage());
                    e.printStackTrace();
                }
            }
        } catch (IOException e) {
            System.err.println("Could not listen on port " + port + ": " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Parse command line arguments
     */
    private static CommandLine parseCommandLineArgs(String[] args) {
        Options options = new Options();

        // Add option for port number
        Option portOption = Option.builder("p").longOpt("port").hasArg(true).desc("Port number for the server (default: " + DEFAULT_PORT + ")").required(false).build();

        // Add option for search depth
        Option depthOption = Option.builder("d").longOpt("depth").hasArg(true).desc("Search depth (default: " + searchDepth + ")").required(false).build();

        // Add option for time limit
        Option timeLimitOption = Option.builder("t").longOpt("time-limit").hasArg(true).desc("Time limit in milliseconds (default: " + timeLimit + ")").required(false).build();

        options.addOption(portOption);
        options.addOption(depthOption);
        options.addOption(timeLimitOption);

        // Add help option
        options.addOption("h", "help", false, "Print this help message");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);

            // Show help if requested
            if (cmd.hasOption("h")) {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("IDAlphaBetaServer", options);
                System.exit(0);
            }
        } catch (ParseException e) {
            System.err.println("Error parsing command line arguments: " + e.getMessage());
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("IDAlphaBetaServer", options);
            System.exit(1);
        }

        return cmd;
    }

    private static KingAndCourtesanBoard createBoardFromJson(JSONArray boardArray) {
        // Create a new board
        int boardSize = boardArray.length();
        KingAndCourtesanBoard board = new KingAndCourtesanBoard(boardSize);

        // We'll need to reconstruct the board manually from the JSON representation
        // First, create an empty board
        KingAndCourtesanBoard emptyBoard = new KingAndCourtesanBoard(boardSize);

        // Parse the pieces
        Point redKingPos = null;
        Point blueKingPos = null;

        // Create board representation string
        StringBuilder boardStr = new StringBuilder();

        // Add the piece positions to the board string
        for (int i = boardSize - 1; i >= 0; i--) {
            JSONArray row = boardArray.getJSONArray(i);
            for (int j = 0; j < row.length(); j++) {
                String pieceStr = row.getString(j);

                if (pieceStr.equals("RED_KING")) {
                    boardStr.append("R");
                    redKingPos = new Point(i, j);
                } else if (pieceStr.equals("RED_COURTESAN")) {
                    boardStr.append("R");
                } else if (pieceStr.equals("BLUE_KING")) {
                    boardStr.append("B");
                    blueKingPos = new Point(i, j);
                } else if (pieceStr.equals("BLUE_COURTESAN")) {
                    boardStr.append("B");
                } else {
                    boardStr.append("-");
                }
            }
            boardStr.append("\n");
        }

        // Add king positions
        if (blueKingPos != null) {
            boardStr.append("BLUE KING Position: (").append(blueKingPos.x).append(",").append(blueKingPos.y).append(")\n");
        }
        if (redKingPos != null) {
            boardStr.append("RED KING Position: (").append(redKingPos.x).append(",").append(redKingPos.y).append(")");
        }

        // Create board from string representation
        return new KingAndCourtesanBoard(boardStr.toString());
    }

    /**
     * Client handler class to process requests for each connected client
     */
    private static class ClientHandler implements Runnable {
        private final Socket clientSocket;
        private GameAlgorithm<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard> redAlgorithm;
        private GameAlgorithm<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard> blueAlgorithm;
        private KingAndCourtesanBoard board;
        private KingAndCourtesanRole currentRole;
        private KingAndCourtesanRole aiRole;

        public ClientHandler(Socket socket) {
            this.clientSocket = socket;
            this.board = new KingAndCourtesanBoard(6);
            this.currentRole = KingAndCourtesanRole.RED;
            this.aiRole = KingAndCourtesanRole.BLUE;

            // Create persistent algorithm instances for this client
            initializeAlgorithms();
        }

        private void initializeAlgorithms() {
            System.out.println("Initializing Alpha-Beta algorithms with depth=" + searchDepth + ", timeLimit=" + timeLimit);

            // Use standard version
            redAlgorithm = new IDAlphaBeta<>(KingAndCourtesanRole.RED, KingAndCourtesanRole.BLUE, KingAndCourtesanHeuristics.hRed, searchDepth, timeLimit);

            blueAlgorithm = new IDAlphaBeta<>(KingAndCourtesanRole.BLUE, KingAndCourtesanRole.RED, KingAndCourtesanHeuristics.hBlue, searchDepth, timeLimit);

            System.out.println("Using IDAlphaBeta algorithm");
        }


        @Override
        public void run() {
            try (BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream())); PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)) {
                String inputLine;
                while ((inputLine = in.readLine()) != null) {
                    try {
                        JSONObject command = new JSONObject(inputLine);
                        JSONObject response = handleCommand(command);
                        out.println(response);
                    } catch (Exception e) {
                        e.printStackTrace();
                        JSONObject errorResponse = new JSONObject();
                        errorResponse.put("error", e.getMessage());
                        out.println(errorResponse);
                    }
                }
            } catch (Exception e) {
                System.err.println("Error handling client: " + e.getMessage());
                e.printStackTrace();
            } finally {
                try {
                    clientSocket.close();
                    System.out.println("Client connection closed");
                } catch (IOException e) {
                    System.err.println("Error closing client socket: " + e.getMessage());
                }
            }
        }

        private JSONObject handleCommand(JSONObject command) throws Exception {
            String cmd = command.getString("command");
            JSONObject response = new JSONObject();

            switch (cmd) {
                case "RESET":
                case "RESET_ID_ALPHA_BETA":
                    // Reset the board state
                    board = new KingAndCourtesanBoard(6);
                    currentRole = KingAndCourtesanRole.RED;

                    // Determine AI role based on first player setting
                    boolean isFirstPlayer = command.optBoolean("is_first_player", true);
                    aiRole = isFirstPlayer ? KingAndCourtesanRole.BLUE : KingAndCourtesanRole.RED;

                    // Return board state and legal moves
                    response.put("board", getBoardAsJson());
                    response.put("legal_moves", getLegalMovesAsJson(currentRole));
                    response.put("current_role", currentRole.toString());
                    response.put("ai_role", aiRole.toString());
                    break;

                case "MOVE":
                    // Process move command
                    String moveStr = command.getString("move");
                    String roleStr = command.getString("role");
                    KingAndCourtesanRole role = KingAndCourtesanRole.valueOf(roleStr);

                    // Create move from string
                    KingAndCourtesanMove move = new KingAndCourtesanMove(moveStr);

                    // Validate move
                    boolean isValid = board.isValidMove(move, role);
                    response.put("valid_move", isValid);

                    if (isValid) {
                        // Execute move
                        board = board.play(move, role);

                        // Switch role
                        currentRole = (role == KingAndCourtesanRole.RED) ? KingAndCourtesanRole.BLUE : KingAndCourtesanRole.RED;

                        // Check game state
                        boolean gameOver = board.isGameOver();
                        response.put("game_over", gameOver);

                        if (gameOver) {
                            if (board.redWins()) {
                                response.put("winner", "RED");
                            } else {
                                response.put("winner", "BLUE");
                            }
                        }

                        // Return updated board and legal moves
                        response.put("board", getBoardAsJson());
                        response.put("legal_moves", getLegalMovesAsJson(currentRole));
                        response.put("current_role", currentRole.toString());
                    }
                    break;

                case "GET_BEST_MOVE":
                    // Parse board state from JSON if provided
                    if (command.has("board")) {
                        JSONArray boardArray = command.getJSONArray("board");
                        board = createBoardFromJson(boardArray);
                    }

                    // Get role from request
                    String requestedRoleStr = command.optString("role", aiRole.toString());
                    KingAndCourtesanRole requestedRole = KingAndCourtesanRole.valueOf(requestedRoleStr);

                    // Choose the appropriate algorithm instance based on role
                    GameAlgorithm<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard> algorithm = (requestedRole == KingAndCourtesanRole.RED) ? redAlgorithm : blueAlgorithm;

                    // Get best move
                    long startTime = System.currentTimeMillis();
                    System.out.println("Computing best move for " + requestedRole + "...");
                    KingAndCourtesanMove bestMove = algorithm.bestMove(board, requestedRole);
                    long endTime = System.currentTimeMillis();

                    long computeTime = endTime - startTime;
                    System.out.println("Found best move: " + bestMove + " (in " + computeTime + "ms)");

                    // Return best move
                    response.put("best_move", bestMove.toString());
                    response.put("computation_time_ms", computeTime);
                    response.put("role", requestedRole.toString());
                    break;

                case "RENDER":
                    response.put("board_string", board.toString());
                    break;

                case "SET_PARAMETERS":
                    // Dynamic parameter updates
                    if (command.has("depth")) {
                        searchDepth = command.getInt("depth");
                        response.put("depth", searchDepth);
                    }
                    if (command.has("time_limit")) {
                        timeLimit = command.getLong("time_limit");
                        response.put("time_limit", timeLimit);
                    }
                    if (command.has("use_transposition_table")) {
                        useTranspositionTable = command.getBoolean("use_transposition_table");
                        response.put("use_transposition_table", useTranspositionTable);
                    }

                    // Reinitialize algorithms with new parameters
                    initializeAlgorithms();
                    response.put("status", "parameters updated");
                    break;

                case "CLOSE":
                    response.put("status", "closed");
                    break;

                default:
                    throw new IllegalArgumentException("Unknown command: " + cmd);
            }

            return response;
        }

        private JSONArray getBoardAsJson() {
            KingAndCourtesanBoard.SQUARE[][] boardGrid = board.getBoardGrid();
            JSONArray boardArray = new JSONArray();

            for (int i = 0; i < boardGrid.length; i++) {
                JSONArray rowArray = new JSONArray();
                for (int j = 0; j < boardGrid[i].length; j++) {
                    rowArray.put(boardGrid[i][j].toString());
                }
                boardArray.put(rowArray);
            }

            return boardArray;
        }

        private JSONArray getLegalMovesAsJson(KingAndCourtesanRole role) {
            ArrayList<KingAndCourtesanMove> legalMoves = board.possibleMoves(role);
            JSONArray movesArray = new JSONArray();

            for (KingAndCourtesanMove move : legalMoves) {
                movesArray.put(move.toString());
            }

            return movesArray;
        }
    }
}