package games.kac;

import iialib.games.algs.GameAlgorithm;
import iialib.games.algs.algorithms.IDAlphaBeta;
import org.apache.commons.cli.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

/**
 * Server for King and Courtesan game that handles client connections
 * and game state management.
 */
public class RL_Agents_KingAndCourtesanServer {
    private static final int DEFAULT_PORT = 42;
    private static final int DEFAULT_BOARD_SIZE = 6;
    private static final int DEFAULT_THREAD_POOL_SIZE = 10;
    private static final int DEFAULT_CLIENT_TIMEOUT = 60; // seconds
    // Logger setup
    private static final Logger logger = Logger.getLogger(RL_Agents_KingAndCourtesanServer.class.getName());
    private static int boardSize = DEFAULT_BOARD_SIZE;
    private static int threadPoolSize = DEFAULT_THREAD_POOL_SIZE;
    private static int clientTimeout = DEFAULT_CLIENT_TIMEOUT;
    private static boolean verboseLogging = false;
    private static ExecutorService threadPool;
    private static ServerSocket serverSocket;
    private static boolean serverRunning = true;

    public static void main(String[] args) {
        // Set up logger
        setupLogger();

        // Parse command line arguments
        CommandLine cmd = parseCommandLineArgs(args);

        // Get port number from arguments or use default
        int port = DEFAULT_PORT;
        if (cmd.hasOption('p')) {
            try {
                port = Integer.parseInt(cmd.getOptionValue('p'));
            } catch (NumberFormatException e) {
                logger.warning("Invalid port number, using default port " + DEFAULT_PORT);
            }
        }

        // Get board size if specified
        if (cmd.hasOption('b')) {
            try {
                boardSize = Integer.parseInt(cmd.getOptionValue('b'));
                if (boardSize < 3 || boardSize > 12) {
                    logger.warning("Board size must be between 3 and 12, using default " + DEFAULT_BOARD_SIZE);
                    boardSize = DEFAULT_BOARD_SIZE;
                }
            } catch (NumberFormatException e) {
                logger.warning("Invalid board size, using default " + DEFAULT_BOARD_SIZE);
            }
        }

        // Get thread pool size if specified
        if (cmd.hasOption('t')) {
            try {
                threadPoolSize = Integer.parseInt(cmd.getOptionValue('t'));
            } catch (NumberFormatException e) {
                logger.warning("Invalid thread pool size, using default " + DEFAULT_THREAD_POOL_SIZE);
            }
        }

        // Get client timeout if specified
        if (cmd.hasOption('c')) {
            try {
                clientTimeout = Integer.parseInt(cmd.getOptionValue('c'));
            } catch (NumberFormatException e) {
                logger.warning("Invalid client timeout, using default " + DEFAULT_CLIENT_TIMEOUT);
            }
        }

        // Check if verbose logging is enabled
        if (cmd.hasOption('v')) {
            verboseLogging = true;
            logger.setLevel(Level.FINE);
        }

        // Create thread pool
        threadPool = Executors.newFixedThreadPool(threadPoolSize);

        // Register shutdown hook for clean shutdown
        registerShutdownHook();

        try {
            // Create server socket
            serverSocket = new ServerSocket(port);
            logger.info("King and Courtesan server started on port " + port);
            logger.info("Configuration: boardSize=" + boardSize + ", threads=" + threadPoolSize +
                    ", clientTimeout=" + clientTimeout + "s, verbose=" + verboseLogging);

            // Main server loop - accept and handle client connections
            while (serverRunning) {
                try {
                    Socket clientSocket = serverSocket.accept();
                    logger.info("Client connected from " + clientSocket.getInetAddress());

                    // Set client socket timeout
                    clientSocket.setSoTimeout(clientTimeout * 1000);

                    // Submit client to thread pool
                    threadPool.submit(new ClientHandler(clientSocket));

                } catch (IOException e) {
                    if (serverRunning) {
                        logger.log(Level.WARNING, "Error accepting client connection", e);
                    }
                }
            }
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Could not listen on port " + port, e);
        } finally {
            shutdownServer();
        }
    }

    /**
     * Set up the logger with file and console output
     */
    private static void setupLogger() {
        try {
            // Create a file handler for logging to a file
            FileHandler fileHandler = new FileHandler("kac_server.log", true);
            fileHandler.setFormatter(new SimpleFormatter());
            logger.addHandler(fileHandler);

            // Set the default level
            logger.setLevel(Level.INFO);
        } catch (IOException e) {
            System.err.println("Failed to set up logger: " + e.getMessage());
        }
    }

    /**
     * Register a shutdown hook for clean server shutdown
     */
    private static void registerShutdownHook() {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Shutdown signal received, stopping server...");
            shutdownServer();
        }));
    }

    /**
     * Shutdown the server cleanly
     */
    private static void shutdownServer() {
        serverRunning = false;

        // Close server socket
        if (serverSocket != null && !serverSocket.isClosed()) {
            try {
                serverSocket.close();
                logger.info("Server socket closed");
            } catch (IOException e) {
                logger.log(Level.WARNING, "Error closing server socket", e);
            }
        }

        // Shutdown thread pool
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
            try {
                // Wait for existing tasks to terminate
                if (!threadPool.awaitTermination(30, TimeUnit.SECONDS)) {
                    threadPool.shutdownNow();
                    if (!threadPool.awaitTermination(30, TimeUnit.SECONDS)) {
                        logger.warning("Thread pool did not terminate");
                    }
                }
                logger.info("Thread pool shutdown complete");
            } catch (InterruptedException e) {
                threadPool.shutdownNow();
                Thread.currentThread().interrupt();
                logger.log(Level.WARNING, "Thread pool shutdown interrupted", e);
            }
        }

        logger.info("Server shutdown complete");
    }

    /**
     * Parse command line arguments
     */
    private static CommandLine parseCommandLineArgs(String[] args) {
        Options options = new Options();

        // Add option for port number
        Option portOption = Option.builder("p")
                .longOpt("port")
                .hasArg(true)
                .desc("Port number for the server (default: " + DEFAULT_PORT + ")")
                .required(false)
                .build();

        // Add option for board size
        Option boardSizeOption = Option.builder("b")
                .longOpt("board-size")
                .hasArg(true)
                .desc("Size of the board (default: " + DEFAULT_BOARD_SIZE + ")")
                .required(false)
                .build();

        // Add option for thread pool size
        Option threadPoolOption = Option.builder("t")
                .longOpt("threads")
                .hasArg(true)
                .desc("Size of the thread pool (default: " + DEFAULT_THREAD_POOL_SIZE + ")")
                .required(false)
                .build();

        // Add option for client timeout
        Option timeoutOption = Option.builder("c")
                .longOpt("client-timeout")
                .hasArg(true)
                .desc("Client socket timeout in seconds (default: " + DEFAULT_CLIENT_TIMEOUT + ")")
                .required(false)
                .build();

        options.addOption(portOption);
        options.addOption(boardSizeOption);
        options.addOption(threadPoolOption);
        options.addOption(timeoutOption);

        // Add help option
        options.addOption("h", "help", false, "Print this help message");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);

            // Show help if requested
            if (cmd.hasOption("h")) {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("RL_Agents_KingAndCourtesanServer", options);
                System.exit(0);
            }
        } catch (ParseException e) {
            System.err.println("Error parsing command line arguments: " + e.getMessage());
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("RL_Agents_KingAndCourtesanServer", options);
            System.exit(1);
        }

        return cmd;
    }

    /**
     * Client handler class to process requests for each connected client
     */
    private static class ClientHandler implements Runnable {
        private final Socket clientSocket;
        private final String clientId;
        private KingAndCourtesanBoard board;
        private KingAndCourtesanRole currentRole;
        private GameAlgorithm<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard> redAlgorithm;
        private GameAlgorithm<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard> blueAlgorithm;
        private KingAndCourtesanRole aiRole;

        public ClientHandler(Socket socket) {
            this.clientSocket = socket;
            this.clientId = socket.getInetAddress() + ":" + socket.getPort();
            resetGame();
            initializeAlgorithms();
        }

        private void resetGame() {
            board = new KingAndCourtesanBoard(boardSize); // Create board with configured size
            currentRole = KingAndCourtesanRole.RED; // RED always starts
            aiRole = KingAndCourtesanRole.BLUE; // Default AI role
        }

        private void initializeAlgorithms() {
            // Create both algorithm instances during initialization
            redAlgorithm = new IDAlphaBeta<>(
                    KingAndCourtesanRole.RED,
                    KingAndCourtesanRole.BLUE,
                    KingAndCourtesanHeuristics.hRed,
                    8
            );

            blueAlgorithm = new IDAlphaBeta<>(
                    KingAndCourtesanRole.BLUE,
                    KingAndCourtesanRole.RED,
                    KingAndCourtesanHeuristics.hBlue,
                    8
            );
        }

        @Override
        public void run() {
            try (
                    BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                    PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)
            ) {
                String inputLine;
                while ((inputLine = in.readLine()) != null) {
                    try {
                        if (verboseLogging) {
                            logger.fine("Client " + clientId + " request: " + inputLine);
                        }

                        JSONObject command = new JSONObject(inputLine);
                        JSONObject response = handleCommand(command);
                        out.println(response);

                        if (verboseLogging) {
                            logger.fine("Client " + clientId + " response sent");
                        }
                    } catch (Exception e) {
                        logger.log(Level.WARNING, "Error processing client request from " + clientId, e);
                        JSONObject errorResponse = new JSONObject();
                        errorResponse.put("error", e.getMessage());
                        out.println(errorResponse);
                    }
                }
            } catch (IOException e) {
                if (!"Connection reset".equals(e.getMessage()) &&
                        !"Socket closed".equals(e.getMessage())) {
                    logger.log(Level.WARNING, "Error handling client " + clientId, e);
                }
            } finally {
                try {
                    clientSocket.close();
                    logger.info("Client " + clientId + " disconnected");
                } catch (IOException e) {
                    logger.log(Level.WARNING, "Error closing client socket for " + clientId, e);
                }
            }
        }

        public JSONObject handleCommand(JSONObject command) throws Exception {
            String cmd = command.getString("command");
            JSONObject response = new JSONObject();

            switch (cmd) {
                case "RESET":
                    resetGame();
                    response.put("board", getBoardAsJson());
                    response.put("legal_moves", getLegalMovesAsJson(currentRole));
                    response.put("current_role", currentRole.toString());
                    logger.info("Client " + clientId + ": Game reset");
                    break;

                case "RESET_ID_ALPHA_BETA":
                    resetGame();
                    boolean isFirstPlayer = command.optBoolean("is_first_player", true);
                    response.put("board", getBoardAsJson());
                    response.put("legal_moves", getLegalMovesAsJson(currentRole));
                    response.put("current_role", currentRole.toString());

                    // Set the AI role based on player selection
                    aiRole = isFirstPlayer ? KingAndCourtesanRole.BLUE : KingAndCourtesanRole.RED;
                    response.put("ai_role", aiRole.toString());
                    logger.info("Client " + clientId + ": Game reset with AI role " + aiRole);
                    break;

                case "MOVE":
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
                        currentRole = (role == KingAndCourtesanRole.RED) ?
                                KingAndCourtesanRole.BLUE : KingAndCourtesanRole.RED;

                        // Check game state
                        boolean gameOver = board.isGameOver();
                        response.put("game_over", gameOver);

                        if (gameOver) {
                            if (board.redWins()) {
                                response.put("winner", "RED");
                                logger.info("Client " + clientId + ": Game over, RED wins");
                            } else {
                                response.put("winner", "BLUE");
                                logger.info("Client " + clientId + ": Game over, BLUE wins");
                            }
                        }

                        // Return updated board and legal moves
                        response.put("board", getBoardAsJson());
                        response.put("legal_moves", getLegalMovesAsJson(currentRole));
                        response.put("current_role", currentRole.toString());

                        logger.info("Client " + clientId + ": Move " + moveStr + " by " + roleStr +
                                (gameOver ? " (GAME OVER)" : ""));
                    } else {
                        logger.warning("Client " + clientId + ": Invalid move attempt " + moveStr + " by " + roleStr);
                    }
                    break;

                case "RENDER":
                    response.put("board_string", board.toString());
                    break;

                case "GET_BEST_MOVE":
                    // Use the appropriate algorithm based on the requested role
                    String requestedRoleStr = command.optString("role", aiRole.toString());
                    KingAndCourtesanRole requestedRole = KingAndCourtesanRole.valueOf(requestedRoleStr);

                    GameAlgorithm<KingAndCourtesanMove, KingAndCourtesanRole, KingAndCourtesanBoard> algorithm =
                            (requestedRole == KingAndCourtesanRole.RED) ? redAlgorithm : blueAlgorithm;

                    logger.info("Client " + clientId + ": Computing best move for " + requestedRole);

                    // Get best move
                    long startTime = System.currentTimeMillis();
                    KingAndCourtesanMove bestMove = algorithm.bestMove(board, requestedRole);
                    long endTime = System.currentTimeMillis();
                    long computeTime = endTime - startTime;

                    // Return best move
                    response.put("best_move", bestMove.toString());
                    response.put("computation_time_ms", computeTime);
                    response.put("role", requestedRole.toString());

                    logger.info("Client " + clientId + ": Best move computed: " + bestMove +
                            " in " + computeTime + "ms");
                    break;

                case "CLOSE":
                    response.put("status", "closed");
                    logger.info("Client " + clientId + ": Close requested");
                    break;

                case "INFO":
                    // Return server information
                    response.put("server_type", "King and Courtesan Game Server");
                    response.put("board_size", boardSize);
                    response.put("current_role", currentRole.toString());
                    response.put("ai_role", aiRole.toString());
                    response.put("game_over", board.isGameOver());
                    if (board.isGameOver()) {
                        response.put("winner", board.redWins() ? "RED" : "BLUE");
                    }
                    break;

                default:
                    logger.warning("Client " + clientId + ": Unknown command: " + cmd);
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