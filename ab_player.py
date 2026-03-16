#Zijie Zhang, Sep.24/2023
#comment
import reversi_server
import numpy as np
import socket, pickle
from reversi import reversi
import math
import time

# keep move under 5 seconds
TIME_LIMIT = 4.7

# positional weights for stronger board evaluation
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

CORNERS = [(0,0), (0,7), (7,0), (7,7)]
X_SQUARES = {
    (1,1):(0,0), (1,6):(0,7), (6,1):(7,0), (6,6):(7,7)
}
C_SQUARES = {
    (0,1):(0,0), (1,0):(0,0),
    (0,6):(0,7), (1,7):(0,7),
    (6,0):(7,0), (7,1):(7,0),
    (6,7):(7,7), (7,6):(7,7)
}

class TimeUp(Exception):
    pass

# def evaluate(board,prev_turn):
#     return np.sum(board == prev_turn) - np.sum(board == -prev_turn)

# stronger heuristic evaluation:
def evaluate(board, prev_turn):
    opp_turn = -prev_turn

    my_discs = np.sum(board == prev_turn)
    opp_discs = np.sum(board == opp_turn)
    empty_count = np.sum(board == 0)

    # 1. Disc difference
    disc_score = my_discs - opp_discs

    # 2. Positional score
    positional_score = np.sum(POSITION_WEIGHTS[board == prev_turn]) - np.sum(POSITION_WEIGHTS[board == opp_turn])

    # 3. Corner score
    my_corners = sum(1 for x, y in CORNERS if board[x, y] == prev_turn)
    opp_corners = sum(1 for x, y in CORNERS if board[x, y] == opp_turn)
    corner_score = 25 * (my_corners - opp_corners)

    # 4. Mobility
    my_moves = len(get_legal_moves(board, prev_turn))
    opp_moves = len(get_legal_moves(board, opp_turn))
    mobility_score = 0
    if my_moves + opp_moves != 0:
        mobility_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

    # 5. Punish risky X-squares and C-squares if corner not owned
    danger_score = 0
    for (x, y), (cx, cy) in X_SQUARES.items():
        if board[x, y] == prev_turn and board[cx, cy] != prev_turn:
            danger_score -= 12
        elif board[x, y] == opp_turn and board[cx, cy] != opp_turn:
            danger_score += 12

    for (x, y), (cx, cy) in C_SQUARES.items():
        if board[x, y] == prev_turn and board[cx, cy] != prev_turn:
            danger_score -= 8
        elif board[x, y] == opp_turn and board[cx, cy] != opp_turn:
            danger_score += 8

    # 6. Game phase weighting
    if empty_count > 40:
        return 2 * disc_score + 6 * positional_score + 8 * mobility_score + 20 * corner_score + danger_score
    elif empty_count > 15:
        return 4 * disc_score + 5 * positional_score + 6 * mobility_score + 25 * corner_score + danger_score
    else:
        return 10 * disc_score + 2 * positional_score + 3 * mobility_score + 30 * corner_score + danger_score


# helper to get legal moves
def get_legal_moves(board, turn):
    game = reversi()
    game.board = board.copy()
    legal_moves = []
    for i in range(8):
        for j in range(8):
            flips = game.step(i, j, turn, False)
            if flips > 0:
                legal_moves.append((i, j, flips))
    return legal_moves


# helper to apply move
def apply_move(board, move, turn):
    game = reversi()
    game.board = board.copy()
    game.step(move[0], move[1], turn, True)
    return game.board


# helper for move ordering
def move_priority(board, move, turn):
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


# Original alpha-beta
# def alpha_beta(board, depth, alpha, beta, current_turn, root_turn):
#
#     game = reversi()
#     game.board = board.copy()
#
#     legal_moves = []
#     for i in range(8):
#         for j in range(8):
#             if game.step(i, j, current_turn, False) > 0:
#                 legal_moves.append((i, j))
#     legal_moves.sort(
#         key=lambda move: game.step(move[0], move[1], current_turn, False),
#         reverse=True
#     )
#     if depth == 0 or not legal_moves:
#         return evaluate(board, root_turn)
#
#     # Maximizing
#     if current_turn == root_turn:
#         value = -math.inf
#
#         for move in legal_moves:
#             new_game = reversi()
#             new_game.board = board.copy()
#             new_game.step(move[0], move[1], current_turn, True)
#
#             value = max(
#                 value,
#                 alpha_beta(new_game.board,
#                            depth - 1,
#                            alpha,
#                            beta,
#                            -current_turn,
#                            root_turn)
#             )
#
#             alpha = max(alpha, value)
#
#             if alpha >= beta:
#                 break
#
#         return value
#
#     # Minimizing
#     else:
#         value = math.inf
#
#         for move in legal_moves:
#             new_game = reversi()
#             new_game.board = board.copy()
#             new_game.step(move[0], move[1], current_turn, True)
#
#             value = min(
#                 value,
#                 alpha_beta(new_game.board,
#                            depth - 1,
#                            alpha,
#                            beta,
#                            -current_turn,
#                            root_turn)
#             )
#
#             beta = min(beta, value)
#
#             if alpha >= beta:
#                 break
#
#         return value

# Replaced with timed alpha-beta
def alpha_beta(board, depth, alpha, beta, current_turn, root_turn, start_time):
    if time.time() - start_time >= TIME_LIMIT:
        raise TimeUp

    legal_moves = get_legal_moves(board, current_turn)
    legal_moves.sort(key=lambda move: move_priority(board, move, current_turn), reverse=True)

    opponent_moves = get_legal_moves(board, -current_turn)

    if depth == 0 or (not legal_moves and not opponent_moves):
        return evaluate(board, root_turn)

    # handle pass turn
    if not legal_moves:
        return alpha_beta(board, depth, alpha, beta, -current_turn, root_turn, start_time)

    # Maximizing
    if current_turn == root_turn:
        value = -math.inf

        for move in legal_moves:
            new_board = apply_move(board, move, current_turn)

            value = max(
                value,
                alpha_beta(new_board,
                           depth - 1,
                           alpha,
                           beta,
                           -current_turn,
                           root_turn,
                           start_time)
            )

            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value

    # Minimizing
    else:
        value = math.inf

        for move in legal_moves:
            new_board = apply_move(board, move, current_turn)

            value = min(
                value,
                alpha_beta(new_board,
                           depth - 1,
                           alpha,
                           beta,
                           -current_turn,
                           root_turn,
                           start_time)
            )

            beta = min(beta, value)

            if alpha >= beta:
                break

        return value


def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:

        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return
        
        #Debug info
        print(turn)
        print(board)
        '''
        #Local Greedy - Replace with your algorithm
        x = -1
        y = -1
        max = 0
        game.board = board
        for i in range(8):
            for j in range(8):
                cur = game.step(i, j, turn, False)
                if cur > max:
                    max = cur
                    x, y = i, j
        '''
        game.board = board

        # legal_moves = []
        # for i in range(8):
        #     for j in range(8):
        #         if game.step(i,j,turn,False) > 0:
        #             legal_moves.append((i,j))
        # if not legal_moves:
        #     x,y = -1,-1
        # else:
        #     depth = 4
        #     alpha = float("-inf")
        #     beta = float("inf")
        #     optimal_value = float("-inf")
        #     optimal_move = legal_moves[0]
        #     
        #     for move in legal_moves:
        #         new_game = reversi()
        #         new_game.board = board.copy()
        #         new_game.step(move[0],move[1],turn,True)
        #
        #         val = alpha_beta(new_game.board,depth -1 ,alpha,beta,-turn,turn)
        #         if val > optimal_value:
        #             optimal_value = val
        #             optimal_move = move
        #         alpha = max(alpha,optimal_value)
        #     x,y = optimal_move

        # Replaced with iterative deepening + time limit

        legal_moves = get_legal_moves(board, turn)

        if not legal_moves:
            x, y = -1, -1
        else:
            legal_moves.sort(key=lambda move: move_priority(board, move, turn), reverse=True)

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
                        val = alpha_beta(new_board, depth - 1, alpha, beta, -turn, turn, start_time)

                        if val > current_best_value:
                            current_best_value = val
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
            
        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x,y]))
        
if __name__ == '__main__':
    main()