
#camille: i added the aplha beta search and actor critic neural network
# the neural network contributes to board evaluation and move ordering
# while alpha-beta search selects the final move under the time limit.
#NEED TO TRAIN

import socket
import pickle
import time
import os

import numpy as np
import torch
import torch.nn as nn

from reversi import reversi


BOARD_SIZE = 8
TIME_LIMIT = 4.7
INF = 10**9

MODEL_PATH = "data/actor_critic.pth"
USE_ML_VALUE = True
USE_ML_MOVE_ORDERING = True


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


def load_model():
    model = ActorCritic()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print("Loaded ML model:", MODEL_PATH)
    else:
        print("No ML model found. Using heuristic search only.")

    model.eval()
    return model


MODEL = load_model()


CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

X_SQUARES = {
    (1, 1): (0, 0),
    (1, 6): (0, 7),
    (6, 1): (7, 0),
    (6, 6): (7, 7),
}

C_SQUARES = {
    (0, 1): (0, 0), (1, 0): (0, 0),
    (0, 6): (0, 7), (1, 7): (0, 7),
    (6, 0): (7, 0), (7, 1): (7, 0),
    (6, 7): (7, 7), (7, 6): (7, 7),
}

POSITION_WEIGHTS = np.array([
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [ 10,  -2,   0,   0,   0,   0,  -2,  10],
    [  5,  -2,   0,   0,   0,   0,  -2,   5],
    [  5,  -2,   0,   0,   0,   0,  -2,   5],
    [ 10,  -2,   0,   0,   0,   0,  -2,  10],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [100, -20,  10,   5,   5,  10, -20, 100],
], dtype=int)


class TimeUp(Exception):
    pass


def get_state(board, turn):
    flat_board = board.flatten()
    state = np.append(flat_board, turn)
    return torch.tensor(state, dtype=torch.float32)


def get_legal_moves(board, turn):
    game = reversi()
    game.board = board.copy()

    legal_moves = []

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            flips = game.step(x, y, turn, False)

            if flips > 0:
                legal_moves.append((x, y, flips))

    return legal_moves


def apply_move(board, move, turn):
    game = reversi()
    game.board = board.copy()

    x, y = move[0], move[1]
    game.step(x, y, turn, True)

    return game.board


def ml_action_scores(board, turn):
    with torch.no_grad():
        state = get_state(board, turn)
        action_scores, _ = MODEL(state)
        return action_scores.numpy()


def ml_board_value(board, root_turn):
    with torch.no_grad():
        state = get_state(board, root_turn)
        _, value = MODEL(state)

    return float(value.item()) * 100


def move_priority(board, move, turn):
    x, y, flips = move
    score = 0

    if (x, y) in CORNERS:
        score += 100000

    score += POSITION_WEIGHTS[x, y] * 20
    score += flips * 5

    if (x, y) in X_SQUARES:
        cx, cy = X_SQUARES[(x, y)]
        if board[cx, cy] != turn:
            score -= 500

    if (x, y) in C_SQUARES:
        cx, cy = C_SQUARES[(x, y)]
        if board[cx, cy] != turn:
            score -= 250

    if USE_ML_MOVE_ORDERING and os.path.exists(MODEL_PATH):
        scores = ml_action_scores(board, turn)
        score += float(scores[x * 8 + y]) * 10

    return score


def opening_move(board, turn, legal_moves):
    pieces_played = int(np.sum(board != 0))

    if pieces_played > 14:
        return None

    legal_set = {(x, y) for x, y, _ in legal_moves}

    for move in CORNERS:
        if move in legal_set:
            return move

    safe_moves = []

    for x, y, flips in legal_moves:
        if (x, y) in X_SQUARES:
            cx, cy = X_SQUARES[(x, y)]
            if board[cx, cy] != turn:
                continue

        if (x, y) in C_SQUARES:
            cx, cy = C_SQUARES[(x, y)]
            if board[cx, cy] != turn:
                continue

        safe_moves.append((x, y, flips))

    if not safe_moves:
        safe_moves = legal_moves

    preferred_openings = [
        (2, 3), (3, 2), (4, 5), (5, 4),
        (2, 4), (3, 5), (4, 2), (5, 3),
        (2, 2), (2, 5), (5, 2), (5, 5),
    ]

    safe_set = {(x, y) for x, y, _ in safe_moves}

    for move in preferred_openings:
        if move in safe_set:
            return move

    safe_moves.sort(
        key=lambda move: move_priority(board, move, turn),
        reverse=True
    )

    x, y, _ = safe_moves[0]
    return x, y


def heuristic_evaluate(board, root_turn):
    opp_turn = -root_turn

    my_count = int(np.sum(board == root_turn))
    opp_count = int(np.sum(board == opp_turn))
    empties = int(np.sum(board == 0))

    phase = (64 - empties) / 64.0

    piece_score = (my_count - opp_count) * (1 + 12 * phase)

    positional_score = int(
        np.sum(POSITION_WEIGHTS * (board == root_turn))
        - np.sum(POSITION_WEIGHTS * (board == opp_turn))
    )

    corner_score = 0
    for x, y in CORNERS:
        if board[x, y] == root_turn:
            corner_score += 250
        elif board[x, y] == opp_turn:
            corner_score -= 250

    my_moves = len(get_legal_moves(board, root_turn))
    opp_moves = len(get_legal_moves(board, opp_turn))

    mobility_score = 0
    if my_moves + opp_moves != 0:
        mobility_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

    danger_score = 0

    for (x, y), (cx, cy) in X_SQUARES.items():
        if board[cx, cy] == 0:
            if board[x, y] == root_turn:
                danger_score -= 80
            elif board[x, y] == opp_turn:
                danger_score += 80

    for (x, y), (cx, cy) in C_SQUARES.items():
        if board[cx, cy] == 0:
            if board[x, y] == root_turn:
                danger_score -= 40
            elif board[x, y] == opp_turn:
                danger_score += 40

    return (
        piece_score
        + 2.5 * positional_score
        + corner_score
        + 4 * mobility_score
        + danger_score
    )


def evaluate(board, root_turn):
    heuristic_score = heuristic_evaluate(board, root_turn)

    if USE_ML_VALUE and os.path.exists(MODEL_PATH):
        ml_score = ml_board_value(board, root_turn)

        #this is where the actor crit n alpha beta comes in
        return 0.75 * heuristic_score + 0.25 * ml_score

    return heuristic_score


def alpha_beta(board, depth, alpha, beta, current_turn, root_turn, start_time):
    if time.time() - start_time >= TIME_LIMIT:
        raise TimeUp

    legal_moves = get_legal_moves(board, current_turn)
    opponent_moves = get_legal_moves(board, -current_turn)

    if not legal_moves and not opponent_moves:
        final_diff = int(np.sum(board == root_turn) - np.sum(board == -root_turn))

        if final_diff > 0:
            return 100000 + final_diff
        elif final_diff < 0:
            return -100000 + final_diff
        return 0

    if depth == 0:
        return evaluate(board, root_turn)

    if not legal_moves:
        return alpha_beta(
            board,
            depth,
            alpha,
            beta,
            -current_turn,
            root_turn,
            start_time
        )

    legal_moves.sort(
        key=lambda move: move_priority(board, move, current_turn),
        reverse=True
    )

    if current_turn == root_turn:
        value = -INF

        for move in legal_moves:
            new_board = apply_move(board, move, current_turn)

            value = max(
                value,
                alpha_beta(
                    new_board,
                    depth - 1,
                    alpha,
                    beta,
                    -current_turn,
                    root_turn,
                    start_time
                )
            )

            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value

    else:
        value = INF

        for move in legal_moves:
            new_board = apply_move(board, move, current_turn)

            value = min(
                value,
                alpha_beta(
                    new_board,
                    depth - 1,
                    alpha,
                    beta,
                    -current_turn,
                    root_turn,
                    start_time
                )
            )

            beta = min(beta, value)

            if alpha >= beta:
                break

        return value


def choose_move(board, turn):
    legal_moves = get_legal_moves(board, turn)

    if not legal_moves:
        return -1, -1

    book_move = opening_move(board, turn, legal_moves)
    if book_move is not None:
        return book_move

    legal_moves.sort(
        key=lambda move: move_priority(board, move, turn),
        reverse=True
    )

    start_time = time.time()

    best_move = legal_moves[0]
    best_value = -INF
    depth = 1

    while True:
        if time.time() - start_time >= TIME_LIMIT:
            break

        current_best_move = best_move
        current_best_value = -INF

        try:
            alpha = -INF
            beta = INF

            for move in legal_moves:
                if time.time() - start_time >= TIME_LIMIT:
                    raise TimeUp

                new_board = apply_move(board, move, turn)

                value = alpha_beta(
                    new_board,
                    depth - 1,
                    alpha,
                    beta,
                    -turn,
                    turn,
                    start_time
                )

                if value > current_best_value:
                    current_best_value = value
                    current_best_move = move

                alpha = max(alpha, current_best_value)

            best_move = current_best_move
            best_value = current_best_value
            depth += 1

        except TimeUp:
            break

    x, y, _ = best_move
    return x, y


def main():
    game_socket = socket.socket()
    game_socket.connect(("127.0.0.1", 33333))

    while True:
        data = game_socket.recv(65536)
        if not data:
            game_socket.close()
            return

        turn, board = pickle.loads(data)
        if turn == 0:
            game_socket.close()
            return
        x, y = choose_move(board, turn)
        game_socket.send(pickle.dumps([x, y]))


if __name__ == "__main__":
    main()