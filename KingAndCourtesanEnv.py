import socket
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time


class KingAndCourtesanEnv(gym.Env):
    """
    Python wrapper for the Java King and Courtesan game environment.
    Follows Gymnasium interface and communicates with Java server via sockets.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, host='localhost', port=42, board_size=6, render_mode=None, connection_retries=3,
                 retry_delay=2.0):
        super(KingAndCourtesanEnv, self).__init__()
        self.host = host
        self.port = port
        self.board_size = board_size
        self.render_mode = render_mode
        self.connection_retries = connection_retries
        self.retry_delay = retry_delay

        # Socket connection to Java server
        self.socket = None
        self.connect_to_server()

        # Game state tracking
        self.current_player = 0  # 0 for RED, 1 for BLUE
        self.is_first_player = True  # Whether the main agent plays first
        self.legal_moves = []

        # Define action and observation spaces
        # Action space: all possible from-to position pairs
        self.action_space = spaces.Discrete(board_size * board_size * board_size * board_size)

        # Observation space: 6 channels
        # (RED_KING, RED_COURTESAN, BLUE_KING, BLUE_COURTESAN, EMPTY, CURRENT_PLAYER)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6, board_size, board_size), dtype=np.float32)

    def connect_to_server(self):
        """Establish connection to Java server with retries"""
        for attempt in range(self.connection_retries):
            try:
                if self.socket:
                    self.socket.close()

                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                print(f"Connected to Java server at {self.host}:{self.port}")
                return
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{self.connection_retries} failed: {e}")
                if attempt < self.connection_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print("All connection attempts failed")
                    raise

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)

        # Randomly determine if main agent plays first (unless specified in options)
        self.is_first_player = bool(np.random.randint(0, 2)) if options is None else options.get('is_first_player',
                                                                                                 True)

        # Send reset command
        response = self._communicate({"command": "RESET", "is_first_player": self.is_first_player})

        # Parse state
        state = self._parse_board_state(response["board"])
        self.current_player = 0  # Reset to RED
        self.legal_moves = response["legal_moves"]

        info = {"legal_moves": self.legal_moves, "is_first_player": self.is_first_player, "current_player_role": "RED"}

        return state, info

    def step(self, action):
        """Execute action and return next state, reward, done, truncated, info"""
        # Convert action index to move string
        move = self._index_to_move(action)

        # Determine current role
        current_role = "RED" if self.current_player == 0 else "BLUE"

        # Send move command
        response = self._communicate({"command": "MOVE", "move": move, "role": current_role})

        # Check if move was valid
        if not response.get("valid_move", True):
            raise ValueError(
                f"Invalid move {move} attempted for role {current_role}")  # Instead of raising an error, we'll give a negative reward and continue
        # print(f"Warning: Invalid move {move} attempted, applying penalty")
        # return self._parse_board_state(response.get("board", [])), -1.0, False, False, {"illegal_move": True}

        # Parse new state
        next_state = self._parse_board_state(response["board"])

        # Terminal rewards
        game_over = response.get("game_over", False)
        # reward = 0.0

        evaluation = None
        if game_over:
            winner = response.get("winner")
            agent_role = "RED" if self.is_first_player else "BLUE"
            reward = 1.0 if winner == agent_role else -1.0
        else:
            # Optional: Small reward for capturing pieces or progressing toward goal
            # This would require tracking the board state before and after the move
            evaluation = self._get_board_evaluation(current_role)

            # Normalize the evaluation
            # The heuristic can return values in a wide range:
            # - Normal values: typically between -3000 and 3000
            # - Extreme values: ±H_WIN (Integer.MAX_VALUE) for winning/losing positions

            # Capping extreme values
            # if evaluation > 10000:  # Arbitrary large but manageable value
            #    evaluation = 10000
            # elif evaluation < -10000:
            #    evaluation = -10000
            reward = np.tanh(evaluation / 2000.0)  # Normalizing to a range suitable for neural network training
        # reward = evaluation / 10000.0  # Scaling to roughly [-1, 1]

        # Update legal moves
        self.legal_moves = response.get("legal_moves", [])

        # Switch player
        self.current_player = 1 - self.current_player

        # Additional info
        info = {"legal_moves": self.legal_moves, "current_player_role": "BLUE" if self.current_player == 1 else "RED",
                "board_evaluation": evaluation}

       # print(f"Evaluation: {evaluation}, Reward: {reward}")
        return next_state, reward, game_over, False, info

    def _get_board_evaluation(self, role):
        """Requests board evaluation from the server"""
        response = self._communicate({"command": "EVALUATE_BOARD", "role": role})
        if not response.get("board_evaluation", True):
            raise ValueError("Board evaluation not available")
        return float(response.get("evaluation", 0))

    def render(self):
        """Render the current state of the game"""
        if self.render_mode == "human":
            response = self._communicate({"command": "RENDER"})
            print(response.get("board_string", "Board render not available"))

    def get_action_mask(self):
        """
        Return a binary mask of legal actions for the current state.
        Useful for masking illegal actions in Reinforcement Learning.
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)

        # Set 1 for each legal move
        for move_str in self.legal_moves:
            action_idx = self._move_to_index(move_str)
            mask[action_idx] = 1

        return mask

    def close(self):
        """Close the environment"""
        if self.socket:
            try:
                self._communicate({"command": "CLOSE"})
                self.socket.close()
                print("Connection to Java server closed")
            except Exception as e:
                print(f"Error closing connection: {e}")
            finally:
                self.socket = None

    def _communicate(self, command):
        """Send command to Java server and receive response"""
        try:
            # Send command
            message = json.dumps(command) + "\n"
            self.socket.sendall(message.encode())

            # Receive response with better buffering
            response_data = ""
            while True:
                try:
                    # Set a timeout to avoid hanging
                    self.socket.settimeout(5.0)
                    chunk = self.socket.recv(4096).decode()

                    if not chunk:
                        break

                    response_data += chunk

                    # Check if we've received a complete JSON object
                    try:
                        json.loads(response_data)
                        break  # Successfully parsed a complete JSON object
                    except json.JSONDecodeError:
                        # Not a complete JSON object yet, continue receiving
                        continue

                except socket.timeout:
                    print("Socket timeout while waiting for response")
                    break

            # Parse JSON response
            if not response_data:
                raise ConnectionError("Received empty response from server")

            return json.loads(response_data)

        except Exception as e:
            print(f"Communication error: {e}")
            # Try to reconnect
            print("Attempting to reconnect...")
            self.connect_to_server()
            # Re-send the command after reconnection
            return self._communicate(command)

    def _parse_board_state(self, board_data):
        """Convert board data from Java to observation tensor"""
        # Initialize observation tensor (6 channels)
        observation = np.zeros((6, self.board_size, self.board_size), dtype=np.float32)

        # Fill in piece positions
        for i in range(self.board_size):
            for j in range(self.board_size):
                if i >= len(board_data) or j >= len(board_data[i]):
                    # Handle potential inconsistencies in board data
                    continue

                piece = board_data[i][j]

                if piece == "RED_KING":
                    observation[0, i, j] = 1.0
                elif piece == "RED_COURTESAN":
                    observation[1, i, j] = 1.0
                elif piece == "BLUE_KING":
                    observation[2, i, j] = 1.0
                elif piece == "BLUE_COURTESAN":
                    observation[3, i, j] = 1.0
                else:  # EMPTY
                    observation[4, i, j] = 1.0

        # Set current player channel
        observation[5, :, :] = self.current_player

        return observation

    def _index_to_move(self, action_idx):
        """Convert action index to move string (format: 'A0-B1')"""
        # Calculate board size squared
        board_sq = self.board_size * self.board_size

        # Extract source and target indices
        from_idx = action_idx // board_sq
        to_idx = action_idx % board_sq

        # Convert to coordinates
        from_row = from_idx // self.board_size
        from_col = from_idx % self.board_size
        to_row = to_idx // self.board_size
        to_col = to_idx % self.board_size

        # Convert to KingAndCourtesanMove string format
        from_pos = f"{chr(65 + from_row)}{from_col}"
        to_pos = f"{chr(65 + to_row)}{to_col}"

        return f"{from_pos}-{to_pos}"

    def _move_to_index(self, move_str):
        """Convert move string to action index"""
        # Parse move string (format: 'A0-B1')
        parts = move_str.split('-')
        from_pos = parts[0]
        to_pos = parts[1]

        # Extract coordinates
        from_row = ord(from_pos[0]) - 65
        from_col = int(from_pos[1])
        to_row = ord(to_pos[0]) - 65
        to_col = int(to_pos[1])

        # Convert to linear indices
        from_idx = from_row * self.board_size + from_col
        to_idx = to_row * self.board_size + to_col

        # Combine into single action index
        return from_idx * (self.board_size * self.board_size) + to_idx

    def sample_legal_action(self):
        """Sample a random legal action from the current state's legal moves"""
        if not self.legal_moves:
            # If no legal moves, return a random action (this shouldn't happen in a valid game)
            raise RuntimeError("No legal moves available")

        # Choose a random legal move
        move_str = np.random.choice(self.legal_moves)
        return self._move_to_index(move_str)
