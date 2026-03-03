#Zijie Zhang, Sep.24/2023
#comment
import numpy as np
import socket, pickle
from reversi import reversi
import math

def evaluate(board,prev_turn):
    return np.sum(board == prev_turn) - np.sum(board == -prev_turn)
def alpha_beta(board, depth, alpha, beta, current_turn, root_turn):

    game = reversi()
    game.board = board.copy()

    legal_moves = []
    for i in range(8):
        for j in range(8):
            if game.step(i, j, current_turn, False) > 0:
                legal_moves.append((i, j))

    if depth == 0 or not legal_moves:
        return evaluate(board, root_turn)

    # Maximizing
    if current_turn == root_turn:
        value = -math.inf

        for move in legal_moves:
            new_game = reversi()
            new_game.board = board.copy()
            new_game.step(move[0], move[1], current_turn, True)

            value = max(
                value,
                alpha_beta(new_game.board,
                           depth - 1,
                           alpha,
                           beta,
                           -current_turn,
                           root_turn)
            )

            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value

    # Minimizing
    else:
        value = math.inf

        for move in legal_moves:
            new_game = reversi()
            new_game.board = board.copy()
            new_game.step(move[0], move[1], current_turn, True)

            value = min(
                value,
                alpha_beta(new_game.board,
                           depth - 1,
                           alpha,
                           beta,
                           -current_turn,
                           root_turn)
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

        legal_moves = []
        for i in range(8):
            for j in range(8):
                if game.step(i,j,turn,False) > 0:
                    legal_moves.append((i,j))
        if not legal_moves:
            x,y = -1,-1
        else:
            depth = 3
            alpha = float("-inf")
            beta = float("inf")
            optimal_value = float("-inf")
            optimal_move = legal_moves[0]
            
            for move in legal_moves:
                new_game = reversi()
                new_game.board = board.copy()
                new_game.step(move[0],move[1],turn,True)

                val = alpha_beta(new_game.board,depth -1 ,alpha,beta,-turn,turn)
                if val > optimal_value:
                    optimal_value = val
                    optimal_move = move
                alpha = max(alpha,optimal_value)
            x,y = optimal_move
            
        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x,y]))
        
if __name__ == '__main__':
    main()