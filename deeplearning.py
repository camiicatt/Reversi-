import numpy as np
import torch
import torch.nn as nn
import random

from reversi import reversi


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(65, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, 64)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        shared_output = self.shared_layers(state)

        action_scores = self.actor(shared_output)
        board_value = self.critic(shared_output)

        return action_scores, board_value


def get_state(board, turn):
    flat_board = board.flatten()
    state = np.append(flat_board, turn)

    return torch.tensor(state, dtype=torch.float32)


def get_legal_moves(board, turn):
    game = reversi()
    game.board = board.copy()

    legal_moves = []

    for x in range(8):
        for y in range(8):
            result = game.step(x, y, turn, False)

            if result > 0:
                legal_moves.append((x, y))

    return legal_moves


def choose_move(model, board, turn):
    state = get_state(board, turn)
    action_scores, board_value = model(state)
    legal_moves = get_legal_moves(board, turn)

    if len(legal_moves) == 0:
        return -1, -1, None, board_value

    mask = torch.full((64,), -1e9)

    for x, y in legal_moves:
        action_index = x * 8 + y
        mask[action_index] = 0

    masked_scores = action_scores + mask
    probabilities = torch.softmax(masked_scores, dim=0)
    action_distribution = torch.distributions.Categorical(probabilities)
    action = action_distribution.sample()
    log_probability = action_distribution.log_prob(action)
    move_index = action.item()

    x = move_index // 8
    y = move_index % 8

    return x, y, log_probability, board_value


def update_model(optimizer, log_probs, values, final_reward, gamma=0.99):
    """
    Args:
        optimizer:    torch optimizer for the ActorCritic model
        log_probs:    list of log π(a|s) collected during the game
        values:       list of V(s) tensors collected during the game
        final_reward: +1 (win), -1 (loss), or 0 (draw)
        gamma:        discount factor
    """
    if len(log_probs) == 0:
        return 0.0  # no moves were made (edge case)

    # Build discounted returns backwards from the terminal reward.
    returns = []
    R = final_reward
    for _ in reversed(range(len(log_probs))):
        R = gamma * R  # discount first so the final move gets gamma * reward
        returns.insert(0, R)
    # Give the last move the full reward signal
    returns[-1] = final_reward

    returns = torch.tensor(returns, dtype=torch.float32)

    actor_loss = 0.0
    critic_loss = 0.0

    for log_prob, value, G in zip(log_probs, values, returns):
        advantage = G - value.item()

        # Policy gradient: push up probability of actions with positive advantage
        actor_loss += -log_prob * advantage

        # Value regression: critic should predict the discounted return
        critic_loss += nn.functional.mse_loss(value.squeeze(), torch.tensor(G))

    total_loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def compute_reward(board, my_turn):
    """
    Determine the reward from the final board state.
      +1  if we have more pieces
      -1  if opponent has more pieces
       0  if draw
    """
    my_pieces = np.sum(board == my_turn)
    opp_pieces = np.sum(board == -my_turn)

    if my_pieces > opp_pieces:
        return 1.0
    elif opp_pieces > my_pieces:
        return -1.0
    else:
        return 0.0


def play_one_game(model, optimizer, my_turn_side=None):
    """
    Play one full game locally against a random opponent, collect trajectory,
    and update the model at game end.

    Returns (reward, loss) for logging.
    """
    if my_turn_side is None:
        my_turn_side = random.choice([1, -1])

    game = reversi()
    log_probs = []
    values = []

    consecutive_passes = 0

    while True:
        # Check if game is over (two passes in a row or board is full)
        if consecutive_passes >= 2 or np.count_nonzero(game.board == 0) == 0:
            break

        legal_moves = get_legal_moves(game.board, game.turn)

        if len(legal_moves) == 0:
            # Pass turn
            consecutive_passes += 1
            game.turn = -game.turn
            continue

        consecutive_passes = 0

        if game.turn == my_turn_side:
            # Model's turn
            x, y, log_probability, board_value = choose_move(model, game.board, game.turn)
            
            if log_probability is not None:
                log_probs.append(log_probability)
                values.append(board_value)
                
            game.step(x, y, game.turn)
            game.turn = -game.turn
        else:
            # Opponent's turn (Random Agent)
            x, y = random.choice(legal_moves)
            game.step(x, y, game.turn)
            game.turn = -game.turn

    reward = compute_reward(game.board, my_turn_side)
    loss = update_model(optimizer, log_probs, values, reward)

    return reward, loss


def main():
    num_games = 500
    save_interval = 50
    lr = 1e-4

    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Try to load existing weights
    try:
        model.load_state_dict(torch.load("data/actor_critic.pth", weights_only=True))
        print("Loaded existing model weights from data/actor_critic.pth")
    except FileNotFoundError:
        print("No existing weights found — starting fresh.")

    wins, losses, draws = 0, 0, 0

    for game_num in range(1, num_games + 1):
        reward, loss = play_one_game(model, optimizer)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

        print(f"Game {game_num}/{num_games}  |  "
              f"Reward: {reward:+.0f}  |  Loss: {loss:.4f}  |  "
              f"Record: {wins}W-{losses}L-{draws}D")

        # Periodically save weights
        if game_num % save_interval == 0:
            torch.save(model.state_dict(), "data/actor_critic.pth")
            print(f"  → Saved model weights (game {game_num})")

    # Final save
    torch.save(model.state_dict(), "data/actor_critic.pth")
    print(f"\nTraining complete. Final record: {wins}W-{losses}L-{draws}D")
    print("Weights saved to data/actor_critic.pth")


if __name__ == "__main__":
    main()