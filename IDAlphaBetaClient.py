import time
import socket
import json


class IDAlphaBetaAgent:
    """
    Agent that utilizes the Java implemented ID Alpha-Beta algorithm to select moves.
    Communicates with a Java server running the algorithm.
    """

    def __init__(self, env, host='localhost', port=5001, timeout=40, max_retries=3):
        """
        Initialize the ID Alpha-Beta agent.

        Parameters:
        -----------
        env: KingAndCourtesanEnv
            The game environment.
        host : str
            The hostname or IP address of the Java ID Alpha-Beta server.
        port : int
            The port number of the Java ID Alpha-Beta server.
        timeout : int
            Socket timeout in seconds.
        max_retries : int
            Maximum number of connection retry attempts.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.socket = None
        self.connected = self.connect()

        # Only try to reset if connected successfully
        if self.connected:
            try:
                response = self._send_command(
                    {"command": "RESET_ID_ALPHA_BETA", "is_first_player": env.is_first_player})
                print(f"ID Alpha-Beta reset response: {response}")
            except Exception as e:
                print(f"Warning: Could not initialize ID Alpha-Beta server: {e}")
                print("Will fallback to sampling legal moves when needed")

    def connect(self):
        """Establish connection to the Java ID Alpha-Beta server."""
        for attempt in range(self.max_retries):
            try:
                if self.socket:
                    self.socket.close()

                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.timeout)
                self.socket.connect((self.host, self.port))
                print(f"Connected to ID Alpha-Beta server at {self.host}:{self.port}")
                return True
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("All connection attempts failed - will use random legal moves instead")
                    return False
        return False

    def select_action(self, state, env):
        """
        Select the best action for the current state using ID Alpha-Beta.

        Parameters:
        -----------
        state : numpy.ndarray
            The current state of the game.
        env : KingAndCourtesanEnv
            The game environment.

        Returns:
        --------
        int
            The action index to perform.
        """
        if not self.connected:
            # legal_move = np.random.choice(env.legal_moves)
            # action_idx = env._move_to_index(legal_move)
            # print(f"Using random legal move: {legal_move}")
            raise RuntimeError("ID Alpha-Beta agent is not connected to the server")

        try:
            # Convert state to board representation for Java
            board_json = self._state_to_board_json(state)

            # Determine current role
            current_role = "RED" if env.current_player == 0 else "BLUE"

            # Create command for Java server
            command = {"command": "GET_BEST_MOVE", "board": board_json, "role": current_role}

            # Send command and get response
            response = self._send_command(command)

            if "error" in response:
                # print(f"Error from ID Alpha-Beta server: {response['error']}")
                # Fall back to random move if there's an error
                # legal_move = np.random.choice(env.legal_moves)
                # return env._move_to_index(legal_move)
                raise RuntimeError(f"Error from ID Alpha-Beta server: {response['error']}")

            # Get best move and convert to action index
            best_move = response["best_move"]
            computation_time = response.get("computation_time_ms", 0)
            print(f"ID Alpha-Beta move: {best_move} (computed in {computation_time}ms)")

            return env._move_to_index(best_move)

        except Exception as e:
            print(f"Error in ID Alpha-Beta agent: {e}")
            raise  # Fallback to random legal move  # if env.legal_moves:  #    legal_move = np.random.choice(env.legal_moves)  #    action_idx = env._move_to_index(legal_move)  #    print(f"Falling back to random legal move: {legal_move}")  #    return None  # else:  #    print("No legal moves available, using random action")  #    return None

    def _send_command(self, command):
        """
        Send command to Java server and receive response.

        Parameters:
        -----------
        command : dict
            The command to send.

        Returns:
        --------
        dict
            The server's response.
        """
        retries = 0

        while retries < self.max_retries:
            try:
                # Convert command to JSON and send
                command_json = json.dumps(command) + "\n"
                self.socket.sendall(command_json.encode())

                # Receive response
                response_data = ""
                while True:
                    buffer = self.socket.recv(4096).decode()
                    if not buffer:
                        break

                    response_data += buffer

                    # Try parsing to see if we have a complete JSON object
                    try:
                        json.loads(response_data)
                        # If we get here, we have a complete JSON object
                        break
                    except json.JSONDecodeError:
                        # Not a complete object yet, keep receiving
                        continue

                # Parse JSON response
                if not response_data:
                    raise ConnectionError("Received empty response from server")

                return json.loads(response_data)

            except Exception as e:
                print(f"Error in communication with ID Alpha-Beta server: {e}")
                retries += 1

                if retries < self.max_retries:
                    print(f"Retrying communication... ({retries}/{self.max_retries})")
                    # Try to reconnect
                    self.connected = self.connect()
                    time.sleep(1)
                else:
                    raise

    def _state_to_board_json(self, state):
        """
        Convert observation tensor to board JSON for Java.

        Parameters:
        -----------
        state : numpy.ndarray
            The state tensor with shape (6, board_size, board_size).

        Returns:
        --------
        list
            2D array representation of the board for JSON.
        """
        board_size = state.shape[1]  # Should be 6
        board_json = []

        for i in range(board_size):
            row = []
            for j in range(board_size):
                if state[0, i, j] == 1:
                    piece = "RED_KING"
                elif state[1, i, j] == 1:
                    piece = "RED_COURTESAN"
                elif state[2, i, j] == 1:
                    piece = "BLUE_KING"
                elif state[3, i, j] == 1:
                    piece = "BLUE_COURTESAN"
                else:
                    piece = "EMPTY"
                row.append(piece)
            board_json.append(row)

        return board_json

    def close(self):
        """Close the connection to the Java server."""
        if self.socket and self.connected:
            try:
                self._send_command({"command": "CLOSE"})
                self.socket.close()
                print("Closed connection to ID Alpha-Beta server")
            except Exception as e:
                print(f"Error closing connection: {e}")
            finally:
                self.socket = None
                self.connected = False
