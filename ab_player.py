#camille cleaned up ab player 
#mar 16 2026

import math
import pickle
import socket
import time

import numpy as np
import reversi_server
from reversi import reversi


TIME_LIMIT = 4.7

POSITION_WEIGHTS = np.array([
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120]
])

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


class TimeUp(Exception):
    """search exceeded 5 seconds"""
    pass


def get_legal_moves(board, turn):
    """returning a list of legal moves as (row, col, flips)."""
    game = reversi()
    game.board = board.copy()

    legal_moves = []
    for i in range(8):
        for j in range(8):
            flips = game.step(i, j, turn, False)
            if flips > 0:
                legal_moves.append((i, j, flips))

    return legal_moves


def apply_move(board, move, turn):
    game = reversi()
    game.board = board.copy()
    game.step(move[0], move[1], turn, True)
    return game.board


def move_priority(board, move, turn):
    """alpha-beta search"""
    x, y, flips = move
    score = 0

    if (x, y) in CORNERS:
        score += 10000

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

    return score


def evaluate(board, root_turn):
    opp_turn = -root_turn

    my_discs = np.sum(board == root_turn)
    opp_discs = np.sum(board == opp_turn)
    empty_count = np.sum(board == 0)

    disc_score = my_discs - opp_discs

    positional_score = (
        np.sum(POSITION_WEIGHTS[board == root_turn]) -
        np.sum(POSITION_WEIGHTS[board == opp_turn])
    )

    my_corners = sum(1 for x, y in CORNERS if board[x, y] == root_turn)
    opp_corners = sum(1 for x, y in CORNERS if board[x, y] == opp_turn)
    corner_score = 25 * (my_corners - opp_corners)

    my_moves = len(get_legal_moves(board, root_turn))
    opp_moves = len(get_legal_moves(board, opp_turn))
    mobility_score = 0
    if my_moves + opp_moves != 0:
        mobility_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

    danger_score = 0

    for (x, y), (cx, cy) in X_SQUARES.items():
        if board[x, y] == root_turn and board[cx, cy] != root_turn:
            danger_score -= 12
        elif board[x, y] == opp_turn and board[cx, cy] != opp_turn:
            danger_score += 12

    for (x, y), (cx, cy) in C_SQUARES.items():
        if board[x, y] == root_turn and board[cx, cy] != root_turn:
            danger_score -= 8
        elif board[x, y] == opp_turn and board[cx, cy] != opp_turn:
            danger_score += 8

    if empty_count > 40:
        return (
            2 * disc_score + 6 * positional_score + 8 * mobility_score + 20 * corner_score + danger_score
        )
    elif empty_count > 15:
        return (
            4 * disc_score + 5 * positional_score + 6 * mobility_score + 25 * corner_score + danger_score
        )
    else:
        return (
            10 * disc_score + 2 * positional_score + 3 * mobility_score + 30 * corner_score + danger_score
        )


def alpha_beta(board, depth, alpha, beta, current_turn, root_turn, start_time):
    """5 sec alpha-beta search"""
    if time.time() - start_time >= TIME_LIMIT:
        raise TimeUp

    legal_moves = get_legal_moves(board, current_turn)
    legal_moves.sort(
        key=lambda move: move_priority(board, move, current_turn),
        reverse=True
    )

    opponent_moves = get_legal_moves(board, -current_turn)

    if not legal_moves and not opponent_moves:
        final_diff = np.sum(board == root_turn) - np.sum(board == -root_turn)
        return 100000 + final_diff if final_diff > 0 else -100000 + final_diff if final_diff < 0 else 0

    if depth == 0:
        return evaluate(board, root_turn)

    if not legal_moves:
        return alpha_beta(
            board, depth, alpha, beta, -current_turn, root_turn, start_time
        )

    if current_turn == root_turn:
        value = -math.inf
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

    value = math.inf
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

    legal_moves.sort(
        key=lambda move: move_priority(board, move, turn),
        reverse=True
    )

    start_time = time.time()
    best_move = legal_moves[0]
    best_value = float("-inf")
    depth = 1

    while True:
        if time.time() - start_time >= TIME_LIMIT:
            break

        current_best_move = best_move
        current_best_value = float("-inf")

        try:
            alpha = float("-inf")
            beta = float("inf")

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
    print("Depth reached:", depth - 1)
    print("Chosen move:", (x, y))
    print("Evaluation:", best_value)
    return x, y


def main():
    game_socket = socket.socket()
    game_socket.connect(("127.0.0.1", 33333))

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        print(turn)
        print(board)

        x, y = choose_move(board, turn)
        game_socket.send(pickle.dumps([x, y]))


if __name__ == "__main__":
    main()
