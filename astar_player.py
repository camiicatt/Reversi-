# LET IT COOOKKK OMGGG, looks reallt bad at first but it picks up i swear 

# win to loose 1:0 

import numpy as np
import socket
import pickle
from reversi import reversi

BOARD_SIZE = 8
INF = 10**9

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

CORNER_ADJ = [
    (0, 1), (1, 0), (1, 1),
    (0, 6), (1, 7), (1, 6),
    (6, 0), (7, 1), (6, 1),
    (6, 7), (7, 6), (6, 6)
]

#  weights i think are ok
POS_W = np.array([
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [ 10,  -2,   0,   0,   0,   0,  -2,  10],
    [  5,  -2,   0,   0,   0,   0,  -2,   5],
    [  5,  -2,   0,   0,   0,   0,  -2,   5],
    [ 10,  -2,   0,   0,   0,   0,  -2,  10],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [100, -20,  10,   5,   5,  10, -20, 100],
], dtype=int)


def valid_moves(board: np.ndarray, turn: int):
    g = reversi()
    g.board = board
    moves = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
        
            if g.step(i, j, turn, False) > 0:
                moves.append((i, j))
    return moves


def apply_move(board: np.ndarray, turn: int, move):
    g2 = reversi()
    g2.board = board.copy()
    x, y = move

    before = g2.board.copy()


    try:
        g2.step(x, y, turn, True)
        if np.array_equal(g2.board, before):
            g2.board = before.copy()
            g2.step(x, y, turn, False)
    except TypeError:
        g2.step(x, y, turn)

    return g2.board


def count_moves(board: np.ndarray, turn: int) -> int:
    return len(valid_moves(board, turn))


def evaluate(board: np.ndarray, me: int) -> float:
    my_count = int(np.sum(board == me))
    opp_count = int(np.sum(board == -me))

    empties = int(np.sum(board == 0))
    phase = (64 - empties) / 64.0  

    piece_score = (my_count - opp_count) * (2 + 10 * phase)

    #score based on where it is

    pos_score = int(np.sum(POS_W * (board == me)) - np.sum(POS_W * (board == -me)))

    corner_score = 0
    for (x, y) in CORNERS:
        if board[x, y] == me:
            corner_score += 300
        elif board[x, y] == -me:
            corner_score -= 300

    my_mob = count_moves(board, me)
    opp_mob = count_moves(board, -me)
    mob_score = 25 * (my_mob - opp_mob)

    adj_penalty = 0
    for (x, y) in CORNER_ADJ:
        cx = 0 if x < 4 else 7
        cy = 0 if y < 4 else 7
        if board[cx, cy] == 0:
            if board[x, y] == me:
                adj_penalty -= 80
            elif board[x, y] == -me:
                adj_penalty += 80

    return piece_score + pos_score + corner_score + mob_score + adj_penalty


def alphabeta(board: np.ndarray, turn: int, me: int, depth: int, alpha: float, beta: float):
    """rreturn (score, best_move)."""
    moves = valid_moves(board, turn)

    if not moves:
        if not valid_moves(board, -turn):
            return evaluate(board, me), None
        score, _ = alphabeta(board, -turn, me, depth, alpha, beta)
        return score, None

    if depth == 0:
        return evaluate(board, me), None

    def move_key(m):
        if m in CORNERS:
            return 1_000_000
        b2 = apply_move(board, turn, m)
        return evaluate(b2, me) if turn == me else -evaluate(b2, me)

    moves.sort(key=move_key, reverse=True)

    best_move = moves[0]

    if turn == me:
        best_score = -INF
        for m in moves:
            b2 = apply_move(board, turn, m)
            score, _ = alphabeta(b2, -turn, me, depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score, best_move
    else:
        best_score = INF
        for m in moves:
            b2 = apply_move(board, turn, m)
            score, _ = alphabeta(b2, -turn, me, depth - 1, alpha, beta)
            if score < best_score:
                best_score = score
                best_move = m
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score, best_move


def choose_move(board: np.ndarray, turn: int, depth: int = 4):
    score, move = alphabeta(board, turn, turn, depth, -INF, INF)
    if move is None:
        return (-1, -1)
    return move


def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))

    while True:
        data = game_socket.recv(65536)
        if not data:
            game_socket.close()
            return

        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        x, y = choose_move(board, turn, depth=4)
        game_socket.send(pickle.dumps([x, y]))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        input("Crashed ")
