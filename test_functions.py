import time
import numpy as np
import torch
import KingAndCourtesanEnv as kac
import IDAlphaBetaClient as id_ab_client

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def test_random_vs_alphabeta(env_host='localhost', env_port=42, agent_host='localhost', agent_port=43,
                             render_mode='human', response_timeout=50, delay_between_moves=1.0, max_steps=50,
                             verbose=True):
    """
    Test a random agent against the ID Alpha-Beta agent in the King and Courtesan game.

    Parameters:
    -----------
    env_host : str
        Hostname for the main game environment server
    env_port : int
        Port for the main game environment server
    agent_host : str
        Hostname for the ID Alpha-Beta agent server
    agent_port : int
        Port for the ID Alpha-Beta agent server
    render_mode : str
        Render mode for the environment ('human' to display, None for no display)
    agent_timeout : int
        Timeout in seconds for the ID Alpha-Beta agent
    delay_between_moves : float
        Delay in seconds between moves for better viewing
    max_steps : int
        Maximum number of game steps
    verbose : bool
        Whether to print detailed game progress

    Returns:
    --------
    dict
        Game statistics including winner, number of steps, etc.
    """

    # Store game statistics
    stats = {'winner': None, 'steps': 0, 'random_player_role': None, 'alphabeta_player_role': None, 'final_reward': 0,
             'game_complete': False}

    # Initialize environment and agent
    env = kac.KingAndCourtesanEnv(host=env_host, port=env_port, render_mode=render_mode)
    id_alpha_beta_agent = id_ab_client.IDAlphaBetaAgent(env=env, host=agent_host, port=agent_port,
                                                        timeout=response_timeout)

    try:
        # Reset the environment
        observation, info = env.reset()

        # Determine player roles
        if env.is_first_player:
            random_player_role = "RED"
            alphabeta_player_role = "BLUE"
            if verbose:
                print("Random agent plays first (RED)")
                print("ID Alpha-Beta agent plays second (BLUE)")
        else:
            random_player_role = "BLUE"
            alphabeta_player_role = "RED"
            if verbose:
                print("ID Alpha-Beta agent plays first (RED)")
                print("Random agent plays second (BLUE)")

        stats['random_player_role'] = random_player_role
        stats['alphabeta_player_role'] = alphabeta_player_role

        # Initialize move counter
        move_count = 0

        # First move (move 0)
        if verbose:
            print(f"Move {move_count}")

        if env.is_first_player:
            # Random agent's first move
            if verbose:
                print("Random agent's turn")
            action = env.sample_legal_action()
        else:
            # Alpha-Beta agent's first move
            if verbose:
                print("ID Alpha-Beta agent's turn")
            action = id_alpha_beta_agent.select_action(observation, env)

        # Play the first move
        observation, reward, done, truncated, info = env.step(action)
        if render_mode == 'human':
            env.render()

        move_count += 1

        # Game loop - continue with alternating turns
        while move_count < max_steps and not (done or truncated):
            if verbose:
                print(f"Move {move_count}")
            stats['steps'] = move_count

            # Delay between moves if specified
            if delay_between_moves > 0:
                time.sleep(delay_between_moves)

            # Proper turn alternation - random plays when move_count%2 equals 0 if it played first
            # or when move_count%2 equals 1 if it played second
            random_turn = (move_count % 2 == 0 and env.is_first_player) or (
                    move_count % 2 == 1 and not env.is_first_player)

            if random_turn:
                # Random agent's turn
                if verbose:
                    print("Random agent's turn")
                action = env.sample_legal_action()
            else:
                # Alpha-Beta agent's turn
                if verbose:
                    print("ID Alpha-Beta agent's turn")
                action = id_alpha_beta_agent.select_action(observation, env)

            # Execute the move
            observation, reward, done, truncated, info = env.step(action)
            stats['final_reward'] = reward

            if render_mode == 'human':
                env.render()

            move_count += 1

        # Game is over
        if done:
            stats['game_complete'] = True
            if reward > 0:
                if env.is_first_player:
                    stats['winner'] = 'RED'
                else:
                    stats['winner'] = 'BLUE'
            elif reward < 0:
                if env.is_first_player:
                    stats['winner'] = 'BLUE'
                else:
                    stats['winner'] = 'RED'
            else:
                stats['winner'] = 'TIE'

            if verbose:
                print(f"Game Over! Winner: {stats['winner']}")

    except Exception as e:
        # Handle exceptions
        if verbose:
            print(f"Error during game: {e}")
        stats['error'] = str(e)

    finally:
        # Clean up resources
        env.close()
        id_alpha_beta_agent.close()

    return stats


def test_q_network_agent(env_host='localhost', env_port=42, q_network=None, board_size=6, render_mode='human',
                         q_network_adversary=None, num_episode=1, render=True, delay_between_moves=0.5):
    """
    Test Q-Network agents against each other in the environment.

    Parameters:
    -----------
    env : KingAndCourtesanEnv
        The game environment
    q_network : QNetwork
        The main Q-Network agent
    q_network_adversary : QNetwork
        The adversary Q-Network agent (if None, will use q_network as adversary)
    num_episode : int
        Number of episodes to run
    render : bool
        Whether to render the game board
    delay_between_moves : float
        Delay in seconds between moves for better visualization

    Returns:
    --------
    dict
    Game statistics including winner, number of steps, etc.
    """
    stats = {'winner': None, 'steps': 0, 'main_network_role': None, 'adversary_role': None, 'final_reward': 0,
             'game_complete': False}
    env = kac.KingAndCourtesanEnv(host=env_host, port=env_port, board_size=board_size, render_mode=render_mode)

    # If adversary not provided, use main network
    if q_network_adversary is None:
        q_network_adversary = q_network

    try:

        for episode_id in range(num_episode):
            state, info = env.reset()
            done = False
            episode_reward = 0.0

            # Determine the roles based on is_first_player
            if env.is_first_player:
                main_agent_role = "RED"
                adversary_role = "BLUE"
                print(f"Main agent plays first ({main_agent_role})")
                print(f"Adversary plays second ({adversary_role})")
            else:
                main_agent_role = "BLUE"
                adversary_role = "RED"
                print(f"Adversary plays first ({adversary_role})")
                print(f"Main agent plays second ({main_agent_role})")

            stats['main_network_role'] = main_agent_role
            stats['adversary_role'] = adversary_role

            # Initial player is RED (0)
            current_player = 0
            move_count = 0

            # Game loop
            while not done:
                move_count += 1
                print(f"Move {move_count}, Player: {'RED' if current_player == 0 else 'BLUE'}")
                stats['steps'] = move_count

                # Determine which network to use based on current player and is_first_player
                is_main_agent_turn = (current_player == 0 and env.is_first_player) or (
                        current_player == 1 and not env.is_first_player)
                current_network = q_network if is_main_agent_turn else q_network_adversary

                # Get legal moves
                legal_moves = env.legal_moves

                # Convert state to tensor for neural network
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    # Get Q-values
                    q_values = current_network(state_tensor)

                    # Create legal moves mask
                    mask = torch.ones_like(q_values) * float('-inf')
                    for move in legal_moves:
                        move_idx = env._move_to_index(move)
                        mask[0, move_idx] = 0

                    # Apply mask and get best legal action
                    masked_q_values = q_values + mask
                    action = masked_q_values.argmax(dim=1).item()

                # Print the selected move
                selected_move = env._index_to_move(action)
                print(f"Selected move: {selected_move}")

                # Take step in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                stats['final_reward'] = reward
                done = terminated or truncated

                # Update total reward (from perspective of the main agent)
                # step function already rewards the main agent
                # episode_reward += reward

                # Render if requested
                if render:
                    env.render()
                    time.sleep(delay_between_moves)

                # Prepare for next step
                state = next_state
                current_player = 1 - current_player

            # Game over
            if done:
                stats['game_complete'] = True
                print(f"Game Over!")
                if stats['final_reward'] > 0:
                    stats['winner'] = main_agent_role
                    print(f"Winner: {main_agent_role}")
                elif episode_reward < 0:
                    stats['winner'] = adversary_role
                    print(f"Winner: {adversary_role}")
                else:
                    stats['winner'] = 'TIE'
                    print("Game ended in a tie")

    except Exception as e:
        print(f"Error during game: {e}")
        raise

    finally:
        # Clean up resources
        env.close()

    return stats


def test_q_network_vs_random(env_host='localhost', env_port=42, q_network=None, board_size=6, render_mode='human',
                             num_episodes=1, delay_between_moves=1.0, max_steps=50, first_player_q_network=None,
                             # None for random, True for Q-Network first, False for random first
                             verbose=True):
    """
    Test a Q-Network agent against a random agent in the King and Courtesan game.

    Parameters:
    -----------
    env_host : str
        Hostname for the game environment server
    env_port : int
        Port for the game environment server
    q_network : QNetwork
        The Q-Network agent to test
    board_size : int
        Size of the game board
    render_mode : str
        Render mode ('human' for visual display, None for no display)
    num_episodes : int
        Number of episodes to run
    delay_between_moves : float
        Delay in seconds between moves for better viewing
    max_steps : int
        Maximum number of steps per episode
    first_player_q_network : bool or None
        Whether Q-Network plays first (True), random plays first (False), or random assignment (None)
    verbose : bool
        Whether to print detailed game progress

    Returns:
    --------
    dict
        Statistics about the competition
    """
    # Initialize results dictionary
    results = {'episodes': [], 'q_network_wins': 0, 'random_wins': 0, 'ties': 0, 'total_steps': 0}

    # Initialize environment
    env = kac.KingAndCourtesanEnv(host=env_host, port=env_port, board_size=board_size, render_mode=render_mode)

    try:
        for episode in range(num_episodes):
            # Determine first player for this episode
            if first_player_q_network is None:
                is_q_network_first = bool(np.random.randint(0, 2))
            else:
                is_q_network_first = first_player_q_network

            # Reset environment with specified first player
            observation, info = env.reset(options={'is_first_player': is_q_network_first})

            # Track episode details
            episode_stats = {'episode': episode + 1, 'steps': 0, 'winner': None,
                             'q_network_role': "RED" if is_q_network_first else "BLUE",
                             'random_role': "BLUE" if is_q_network_first else "RED"}

            if verbose:
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Q-Network plays {'first (RED)' if is_q_network_first else 'second (BLUE)'}")
                print(f"Random agent plays {'second (BLUE)' if is_q_network_first else 'first (RED)'}")

            # Game loop variables
            move_count = 0
            done = False
            truncated = False

            # Game loop
            while move_count < max_steps and not (done or truncated):
                # Current player determination (0 = RED, 1 = BLUE)
                current_player = move_count % 2

                # Determine if it's Q-Network's turn
                is_q_network_turn = (current_player == 0 and is_q_network_first) or (
                        current_player == 1 and not is_q_network_first)

                if verbose:
                    print(f"\nMove {move_count + 1}")
                    print(
                        f"Player: {'Q-Network' if is_q_network_turn else 'Random'} ({'RED' if current_player == 0 else 'BLUE'})")

                # Select action based on agent type
                if is_q_network_turn:
                    # Q-Network's turn
                    # Convert state to tensor
                    state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

                    # Get Q-values with gradient disabled
                    with torch.no_grad():
                        q_values = q_network(state_tensor)

                    # Mask illegal actions
                    mask = torch.ones_like(q_values) * float('-inf')
                    for move in env.legal_moves:
                        move_idx = env._move_to_index(move)
                        mask[0, move_idx] = 0

                    # Select best legal action
                    masked_q_values = q_values + mask
                    action = masked_q_values.argmax(dim=1).item()

                else:
                    # Random agent's turn
                    action = env.sample_legal_action()

                # Print the selected move
                selected_move = env._index_to_move(action)
                if verbose:
                    print(f"Selected move: {selected_move}")

                # Execute action in environment
                next_observation, reward, done, truncated, info = env.step(action)

                # Render if requested
                if render_mode == 'human':
                    env.render()
                    if delay_between_moves > 0:
                        time.sleep(delay_between_moves)

                # Update for next iteration
                observation = next_observation
                move_count += 1
                episode_stats['steps'] = move_count

            # Episode complete
            if done:
                # Determine winner based on reward
                episode_stats['final_reward'] = reward

                # Translate reward to winner from environment perspective
                if reward > 0:
                    episode_stats['winner'] = 'q_network'
                    results['q_network_wins'] += 1
                elif reward < 0:
                    episode_stats['winner'] = 'random'
                    results['random_wins'] += 1
                else:  # Tie
                    episode_stats['winner'] = 'tie'
                    results['ties'] += 1

                if verbose:
                    print(f"\nGame Over! Winner: {episode_stats['winner']}")
            else:
                episode_stats['winner'] = 'incomplete'

            # Add episode stats to results
            results['episodes'].append(episode_stats)
            results['total_steps'] += move_count

        # Calculate summary statistics
        results['avg_steps_per_episode'] = results['total_steps'] / num_episodes
        results['q_network_win_rate'] = results['q_network_wins'] / num_episodes
        results['random_win_rate'] = results['random_wins'] / num_episodes
        results['tie_rate'] = results['ties'] / num_episodes

        if verbose:
            print("\nTest Results Summary:")
            print(f"Episodes: {num_episodes}")
            print(f"Q-Network wins: {results['q_network_wins']} ({results['q_network_win_rate'] * 100:.1f}%)")
            print(f"Random agent wins: {results['random_wins']} ({results['random_win_rate'] * 100:.1f}%)")
            print(f"Ties: {results['ties']} ({results['tie_rate'] * 100:.1f}%)")
            print(f"Average steps per episode: {results['avg_steps_per_episode']:.1f}")

        return results

    finally:
        # Clean up resources
        env.close()


def test_q_network_vs_alpha_beta(env_host='localhost', env_port=42, agent_host='localhost', agent_port=43,
                                 q_network=None, board_size=6, render_mode='human', response_timeout=50, num_episodes=1,
                                 delay_between_moves=1.0, max_steps=50, first_player_q_network=None,
                                 # None for random, True for Q-Network first, False for Alpha-Beta first
                                 verbose=True):
    """
    Test a Q-Network agent against an ID Alpha-Beta agent in the King and Courtesan game.

    Parameters:
    -----------
    env_host : str
        Hostname for the game environment server
    env_port : int
        Port for the game environment server
    agent_host : str
        Hostname for the ID Alpha-Beta agent server
    agent_port : int
        Port for the ID Alpha-Beta agent server
    q_network : QNetwork
        The Q-Network agent to test
    board_size : int
        Size of the game board
    render_mode : str
        Render mode ('human' for visual display, None for no display)
    response_timeout : int
        Timeout in seconds for the ID Alpha-Beta agent
    num_episodes : int
        Number of episodes to run
    delay_between_moves : float
        Delay in seconds between moves for better viewing
    max_steps : int
        Maximum number of steps per episode
    first_player_q_network : bool or None
        Whether Q-Network plays first (True), Alpha-Beta plays first (False), or random assignment (None)
    verbose : bool
        Whether to print detailed game progress

    Returns:
    --------
    dict
        Statistics about the competition
    """
    # Initialize results dictionary
    results = {'episodes': [], 'q_network_wins': 0, 'alpha_beta_wins': 0, 'ties': 0, 'total_steps': 0}

    # Initialize environment
    env = kac.KingAndCourtesanEnv(host=env_host, port=env_port, board_size=board_size, render_mode=render_mode)

    # Initialize Alpha-Beta agent
    id_alpha_beta_agent = id_ab_client.IDAlphaBetaAgent(env=env, host=agent_host, port=agent_port,
                                                        timeout=response_timeout)

    try:
        for episode in range(num_episodes):
            # Determine first player for this episode
            if first_player_q_network is None:
                is_q_network_first = bool(np.random.randint(0, 2))
            else:
                is_q_network_first = first_player_q_network

            # Reset environment with specified first player
            observation, info = env.reset(options={'is_first_player': is_q_network_first})

            # Track episode details
            episode_stats = {'episode': episode + 1, 'steps': 0, 'winner': None,
                             'q_network_role': "RED" if is_q_network_first else "BLUE",
                             'alpha_beta_role': "BLUE" if is_q_network_first else "RED"}

            if verbose:
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Q-Network plays {'first (RED)' if is_q_network_first else 'second (BLUE)'}")
                print(f"Alpha-Beta agent plays {'second (BLUE)' if is_q_network_first else 'first (RED)'}")

            # Game loop variables
            move_count = 0
            done = False
            truncated = False

            # Game loop
            while move_count < max_steps and not (done or truncated):
                # Current player determination (0 = RED, 1 = BLUE)
                current_player = move_count % 2

                # Determine if it's Q-Network's turn
                is_q_network_turn = (current_player == 0 and is_q_network_first) or (
                        current_player == 1 and not is_q_network_first)

                if verbose:
                    print(f"\nMove {move_count + 1}")
                    print(
                        f"Player: {'Q-Network' if is_q_network_turn else 'Alpha-Beta'} ({'RED' if current_player == 0 else 'BLUE'})")

                # Select action based on agent type
                if is_q_network_turn:
                    # Q-Network's turn
                    # Convert state to tensor
                    state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

                    # Get Q-values with gradient disabled
                    with torch.no_grad():
                        q_values = q_network(state_tensor)

                    # Mask illegal actions
                    mask = torch.ones_like(q_values) * float('-inf')
                    for move in env.legal_moves:
                        move_idx = env._move_to_index(move)
                        mask[0, move_idx] = 0

                    # Select best legal action
                    masked_q_values = q_values + mask
                    action = masked_q_values.argmax(dim=1).item()

                else:
                    # Alpha-Beta agent's turn
                    action = id_alpha_beta_agent.select_action(observation, env)

                # Print the selected move
                selected_move = env._index_to_move(action)
                if verbose:
                    print(f"Selected move: {selected_move}")

                # Execute action in environment
                next_observation, reward, done, truncated, info = env.step(action)

                # Render if requested
                if render_mode == 'human':
                    env.render()
                    if delay_between_moves > 0:
                        time.sleep(delay_between_moves)

                # Update for next iteration
                observation = next_observation
                move_count += 1
                episode_stats['steps'] = move_count

            # Episode complete
            if done:
                # Determine winner based on reward
                episode_stats['final_reward'] = reward

                # Translate reward to winner from environment perspective
                if reward > 0:
                    if is_q_network_first:
                        episode_stats['winner'] = 'q_network'
                        results['q_network_wins'] += 1
                elif reward < 0:
                    if is_q_network_first:
                        episode_stats['winner'] = 'alpha_beta'
                        results['alpha_beta_wins'] += 1
                else:  # Tie
                    episode_stats['winner'] = 'tie'
                    results['ties'] += 1

                if verbose:
                    print(f"\nGame Over! Winner: {episode_stats['winner']}")
            else:
                episode_stats['winner'] = 'incomplete'

            # Add episode stats to results
            results['episodes'].append(episode_stats)
            results['total_steps'] += move_count

        # Calculate summary statistics
        results['avg_steps_per_episode'] = results['total_steps'] / num_episodes
        results['q_network_win_rate'] = results['q_network_wins'] / num_episodes
        results['alpha_beta_win_rate'] = results['alpha_beta_wins'] / num_episodes
        results['tie_rate'] = results['ties'] / num_episodes

        if verbose:
            print("\nTest Results Summary:")
            print(f"Episodes: {num_episodes}")
            print(f"Q-Network wins: {results['q_network_wins']} ({results['q_network_win_rate'] * 100:.1f}%)")
            print(f"Alpha-Beta wins: {results['alpha_beta_wins']} ({results['alpha_beta_win_rate'] * 100:.1f}%)")
            print(f"Ties: {results['ties']} ({results['tie_rate'] * 100:.1f}%)")
            print(f"Average steps per episode: {results['avg_steps_per_episode']:.1f}")

        return results

    finally:
        # Clean up resources
        env.close()
        id_alpha_beta_agent.close()


def test_policy_network_agent(env_host='localhost', env_port=42, policy_network=None, board_size=6, render_mode='human',
                              policy_network_adversary=None, num_episode=1, render=True, delay_between_moves=0.5):
    """
    Test Policy Network agents against each other in the environment.

    Parameters:
    -----------
    env : KingAndCourtesanEnv
        The game environment
    policy_network : PolicyNetwork
        The main Policy Network agent
    policy_network_adversary : PolicyNetwork
        The adversary Policy Network agent (if None, will use policy_network as adversary)
    num_episode : int
        Number of episodes to run
    render : bool
        Whether to render the game board
    delay_between_moves : float
        Delay in seconds between moves for better visualization

    Returns:
    --------
    dict
        Game statistics including winner, number of steps, etc.
    """
    stats = {'winner': None, 'steps': 0, 'main_network_role': None, 'adversary_role': None, 'final_reward': 0,
             'game_complete': False}

    env = kac.KingAndCourtesanEnv(host=env_host, port=env_port, board_size=board_size, render_mode=render_mode)

    # If adversary not provided, use main network
    if policy_network_adversary is None:
        policy_network_adversary = policy_network

    try:
        for episode_id in range(num_episode):
            state, info = env.reset()
            done = False
            episode_reward = 0.0

            # Determine the roles based on is_first_player
            if env.is_first_player:
                main_agent_role = "RED"
                adversary_role = "BLUE"
                print(f"Main agent plays first ({main_agent_role})")
                print(f"Adversary plays second ({adversary_role})")
            else:
                main_agent_role = "BLUE"
                adversary_role = "RED"
                print(f"Adversary plays first ({adversary_role})")
                print(f"Main agent plays second ({main_agent_role})")

            stats['main_network_role'] = main_agent_role
            stats['adversary_role'] = adversary_role

            # Initial player is RED (0)
            current_player = 0
            move_count = 0

            # Game loop
            while not done:
                move_count += 1
                print(f"Move {move_count}, Player: {'RED' if current_player == 0 else 'BLUE'}")
                stats['steps'] = move_count

                # Determine which network to use based on current player and is_first_player
                is_main_agent_turn = (current_player == 0 and env.is_first_player) or (
                        current_player == 1 and not env.is_first_player)
                current_network = policy_network if is_main_agent_turn else policy_network_adversary

                # Sample action from policy network
                action, _ = current_network.sample_discrete_action(env, state)

                # Print the selected move
                selected_move = env._index_to_move(action)
                print(f"Selected move: {selected_move}")

                # Take step in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                stats['final_reward'] = reward
                done = terminated or truncated

                # Render if requested
                if render:
                    env.render()
                    time.sleep(delay_between_moves)

                # Prepare for next step
                state = next_state
                current_player = 1 - current_player

            # Game over
            if done:
                stats['game_complete'] = True
                print(f"Game Over!")
                if stats['final_reward'] > 0:
                    stats['winner'] = main_agent_role
                    print(f"Winner: {main_agent_role}")
                elif stats['final_reward'] < 0:
                    stats['winner'] = adversary_role
                    print(f"Winner: {adversary_role}")
                else:
                    stats['winner'] = 'TIE'
                    print("Game ended in a tie")

    except Exception as e:
        print(f"Error during game: {e}")
        raise

    finally:
        # Clean up resources
        env.close()

    return stats


def test_policy_network_vs_random(env_host='localhost', env_port=42, policy_network=None, board_size=6,
                                  render_mode='human', num_episodes=1, delay_between_moves=1.0, max_steps=50,
                                  first_player_policy_network=None,
                                  # None for random, True for Policy Network first, False for random first
                                  verbose=True):
    """
    Test a Policy Network agent against a random agent in the King and Courtesan game.

    Parameters:
    -----------
    env_host : str
        Hostname for the game environment server
    env_port : int
        Port for the game environment server
    policy_network : PolicyNetwork
        The Policy Network agent to test
    board_size : int
        Size of the game board
    render_mode : str
        Render mode ('human' for visual display, None for no display)
    num_episodes : int
        Number of episodes to run
    delay_between_moves : float
        Delay in seconds between moves for better viewing
    max_steps : int
        Maximum number of steps per episode
    first_player_policy_network : bool or None
        Whether Policy Network plays first (True), random plays first (False), or random assignment (None)
    verbose : bool
        Whether to print detailed game progress

    Returns:
    --------
    dict
        Statistics about the competition
    """
    # Initialize results dictionary
    results = {'episodes': [], 'policy_network_wins': 0, 'random_wins': 0, 'ties': 0, 'total_steps': 0}

    # Initialize environment
    env = kac.KingAndCourtesanEnv(host=env_host, port=env_port, board_size=board_size, render_mode=render_mode)

    try:
        for episode in range(num_episodes):
            # Determine first player for this episode
            if first_player_policy_network is None:
                is_policy_network_first = bool(np.random.randint(0, 2))
            else:
                is_policy_network_first = first_player_policy_network

            # Reset environment with specified first player
            observation, info = env.reset(options={'is_first_player': is_policy_network_first})

            # Track episode details
            episode_stats = {'episode': episode + 1, 'steps': 0, 'winner': None,
                             'policy_network_role': "RED" if is_policy_network_first else "BLUE",
                             'random_role': "BLUE" if is_policy_network_first else "RED"}

            if verbose:
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Policy Network plays {'first (RED)' if is_policy_network_first else 'second (BLUE)'}")
                print(f"Random agent plays {'second (BLUE)' if is_policy_network_first else 'first (RED)'}")

            # Game loop variables
            move_count = 0
            done = False
            truncated = False

            # Game loop
            while move_count < max_steps and not (done or truncated):
                # Current player determination (0 = RED, 1 = BLUE)
                current_player = move_count % 2

                # Determine if it's Policy Network's turn
                is_policy_network_turn = (current_player == 0 and is_policy_network_first) or (
                        current_player == 1 and not is_policy_network_first)

                if verbose:
                    print(f"\nMove {move_count + 1}")
                    print(f"Player: {'Policy Network' if is_policy_network_turn else 'Random'} "
                          f"({'RED' if current_player == 0 else 'BLUE'})")

                # Select action based on agent type
                if is_policy_network_turn:
                    # Policy Network's turn - sample from distribution
                    action, _ = policy_network.sample_discrete_action(env, observation)
                else:
                    # Random agent's turn
                    action = env.sample_legal_action()

                # Print the selected move
                selected_move = env._index_to_move(action)
                if verbose:
                    print(f"Selected move: {selected_move}")

                # Execute action in environment
                next_observation, reward, done, truncated, info = env.step(action)

                # Render if requested
                if render_mode == 'human':
                    env.render()
                    if delay_between_moves > 0:
                        time.sleep(delay_between_moves)

                # Update for next iteration
                observation = next_observation
                move_count += 1
                episode_stats['steps'] = move_count

            # Episode complete
            if done:
                # Determine winner based on reward
                episode_stats['final_reward'] = reward

                # Translate reward to winner from environment perspective
                if reward > 0:
                    episode_stats['winner'] = 'policy_network'
                    results['policy_network_wins'] += 1
                elif reward < 0:
                    episode_stats['winner'] = 'random'
                    results['random_wins'] += 1
                else:  # Tie
                    episode_stats['winner'] = 'tie'
                    results['ties'] += 1

                if verbose:
                    print(f"\nGame Over! Winner: {episode_stats['winner']}")
            else:
                episode_stats['winner'] = 'incomplete'

            # Add episode stats to results
            results['episodes'].append(episode_stats)
            results['total_steps'] += move_count

        # Calculate summary statistics
        results['avg_steps_per_episode'] = results['total_steps'] / num_episodes
        results['policy_network_win_rate'] = results['policy_network_wins'] / num_episodes
        results['random_win_rate'] = results['random_wins'] / num_episodes
        results['tie_rate'] = results['ties'] / num_episodes

        if verbose:
            print("\nTest Results Summary:")
            print(f"Episodes: {num_episodes}")
            print(
                f"Policy Network wins: {results['policy_network_wins']} ({results['policy_network_win_rate'] * 100:.1f}%)")
            print(f"Random agent wins: {results['random_wins']} ({results['random_win_rate'] * 100:.1f}%)")
            print(f"Ties: {results['ties']} ({results['tie_rate'] * 100:.1f}%)")
            print(f"Average steps per episode: {results['avg_steps_per_episode']:.1f}")

        return results

    finally:
        # Clean up resources
        env.close()


def test_policy_network_vs_alpha_beta(env_host='localhost', env_port=42, agent_host='localhost', agent_port=43,
                                      policy_network=None, board_size=6, render_mode='human', response_timeout=50,
                                      num_episodes=1, delay_between_moves=1.0, max_steps=50,
                                      first_player_policy_network=None,
                                      # None for random, True for Policy Network first, False for Alpha-Beta first
                                      verbose=True):
    """
    Test a Policy Network agent against an ID Alpha-Beta agent in the King and Courtesan game.

    Parameters:
    -----------
    env_host : str
        Hostname for the game environment server
    env_port : int
        Port for the game environment server
    agent_host : str
        Hostname for the ID Alpha-Beta agent server
    agent_port : int
        Port for the ID Alpha-Beta agent server
    policy_network : PolicyNetwork
        The Policy Network agent to test
    board_size : int
        Size of the game board
    render_mode : str
        Render mode ('human' for visual display, None for no display)
    response_timeout : int
        Timeout in seconds for the ID Alpha-Beta agent
    num_episodes : int
        Number of episodes to run
    delay_between_moves : float
        Delay in seconds between moves for better viewing
    max_steps : int
        Maximum number of steps per episode
    first_player_policy_network : bool or None
        Whether Policy Network plays first (True), Alpha-Beta plays first (False), or random assignment (None)
    verbose : bool
        Whether to print detailed game progress

    Returns:
    --------
    dict
        Statistics about the competition
    """
    # Initialize results dictionary
    results = {'episodes': [], 'policy_network_wins': 0, 'alpha_beta_wins': 0, 'ties': 0, 'total_steps': 0}

    # Initialize environment
    env = kac.KingAndCourtesanEnv(host=env_host, port=env_port, board_size=board_size, render_mode=render_mode)

    # Initialize Alpha-Beta agent
    id_alpha_beta_agent = id_ab_client.IDAlphaBetaAgent(env=env, host=agent_host, port=agent_port,
                                                        timeout=response_timeout)

    try:
        for episode in range(num_episodes):
            # Determine first player for this episode
            if first_player_policy_network is None:
                is_policy_network_first = bool(np.random.randint(0, 2))
            else:
                is_policy_network_first = first_player_policy_network

            # Reset environment with specified first player
            observation, info = env.reset(options={'is_first_player': is_policy_network_first})

            # Track episode details
            episode_stats = {'episode': episode + 1, 'steps': 0, 'winner': None,
                             'policy_network_role': "RED" if is_policy_network_first else "BLUE",
                             'alpha_beta_role': "BLUE" if is_policy_network_first else "RED"}

            if verbose:
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Policy Network plays {'first (RED)' if is_policy_network_first else 'second (BLUE)'}")
                print(f"Alpha-Beta agent plays {'second (BLUE)' if is_policy_network_first else 'first (RED)'}")

            # Game loop variables
            move_count = 0
            done = False
            truncated = False

            # Game loop
            while move_count < max_steps and not (done or truncated):
                # Current player determination (0 = RED, 1 = BLUE)
                current_player = move_count % 2

                # Determine if it's Policy Network's turn
                is_policy_network_turn = (current_player == 0 and is_policy_network_first) or (
                        current_player == 1 and not is_policy_network_first)

                if verbose:
                    print(f"\nMove {move_count + 1}")
                    print(f"Player: {'Policy Network' if is_policy_network_turn else 'Alpha-Beta'} "
                          f"({'RED' if current_player == 0 else 'BLUE'})")

                # Select action based on agent type
                if is_policy_network_turn:
                    # Policy Network's turn - sample from distribution
                    action, _ = policy_network.sample_discrete_action(env, observation)
                else:
                    # Alpha-Beta agent's turn
                    action = id_alpha_beta_agent.select_action(observation, env)

                # Print the selected move
                selected_move = env._index_to_move(action)
                if verbose:
                    print(f"Selected move: {selected_move}")

                # Execute action in environment
                next_observation, reward, done, truncated, info = env.step(action)

                # Render if requested
                if render_mode == 'human':
                    env.render()
                    if delay_between_moves > 0:
                        time.sleep(delay_between_moves)

                # Update for next iteration
                observation = next_observation
                move_count += 1
                episode_stats['steps'] = move_count

            # Episode complete
            if done:
                # Determine winner based on reward
                episode_stats['final_reward'] = reward

                # Translate reward to winner from environment perspective
                if reward > 0:
                    episode_stats['winner'] = 'policy_network'
                    results['policy_network_wins'] += 1
                elif reward < 0:
                    episode_stats['winner'] = 'alpha_beta'
                    results['alpha_beta_wins'] += 1
                else:  # Tie
                    episode_stats['winner'] = 'tie'
                    results['ties'] += 1

                if verbose:
                    print(f"\nGame Over! Winner: {episode_stats['winner']}")
            else:
                episode_stats['winner'] = 'incomplete'

            # Add episode stats to results
            results['episodes'].append(episode_stats)
            results['total_steps'] += move_count

        # Calculate summary statistics
        results['avg_steps_per_episode'] = results['total_steps'] / num_episodes
        results['policy_network_win_rate'] = results['policy_network_wins'] / num_episodes
        results['alpha_beta_win_rate'] = results['alpha_beta_wins'] / num_episodes
        results['tie_rate'] = results['ties'] / num_episodes

        if verbose:
            print("\nTest Results Summary:")
            print(f"Episodes: {num_episodes}")
            print(
                f"Policy Network wins: {results['policy_network_wins']} ({results['policy_network_win_rate'] * 100:.1f}%)")
            print(f"Alpha-Beta wins: {results['alpha_beta_wins']} ({results['alpha_beta_win_rate'] * 100:.1f}%)")
            print(f"Ties: {results['ties']} ({results['tie_rate'] * 100:.1f}%)")
            print(f"Average steps per episode: {results['avg_steps_per_episode']:.1f}")

        return results

    finally:
        # Clean up resources
        env.close()
        id_alpha_beta_agent.close()
