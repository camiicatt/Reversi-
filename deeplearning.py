import socket
import pickle
import numpy as np
import torch
import torch.nn as nn

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


def update_model_later():
    """
    Write the reward/training code here.

    For now, this function does nothing.
    """
    pass


def main():
    game_socket = socket.socket()
    game_socket.connect(("127.0.0.1", 33333))

    model = ActorCritic()

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        print("Current turn:", turn)
        print(board)

        x, y, log_probability, board_value = choose_move(model, board, turn)

        print("Chosen move:", x, y)
        print("Critic board value:", board_value.item())

        game_socket.send(pickle.dumps([x, y]))


if __name__ == "__main__":
    main()