# IMPORTS
import ai
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold

# LOAD IN DATASETS

# Final board data for win check
final_data = np.loadtxt("./datasets/tictac_final.txt")
# Split final data into features and lables
final_features = final_data[:,:-1]
final_labels = final_data[:,-1]

# Get Optimal Model
mlp_clf_optimal = ai.train_model()

# TIC TAC TOE GAME

# Description
print('Each cell in the tic-tac-toe Game is represented by a number 1-9. The top 3 cells are 1,2,3, the second row is 4,5,6, and the last row is 7,8,9')
print('The model cells are marked as -1 and your cells will be marked as 1. Empty cells are 0.')

# Checks win, returns True if game is over, False otherwise
def check_win(game_state):

  # Get all rows of final_features that are the game_state
  matching_rows = np.all(final_features == game_state, axis=1)

  # Current board is a winning position
  if np.any(matching_rows):
    # Get first matching row
    index = np.where(matching_rows)[0][0]
    if final_labels[index] == 1:
      print('YOU WIN!!')  # Winner is X (player)
    else:
      print('You lost.')  # Winner is O (AI)
    return True
  return False

# Prints current board
def print_game(game_state):
  # Map for char conversion
  symbol_mapping = {-1: 'O', 0: ' ' , 1: 'X'}
  # Reshape to 3x3 for printing
  game_state = game_state.reshape(3, 3)
  symbol_game = np.vectorize(symbol_mapping.get)(game_state)
  print(symbol_game)
  game_state = game_state.flatten()

# Build and print initial board
game_state = np.zeros(9)
print_game(game_state)

# GAME LOOP
while True:
  # USER MOVE HANDLING
  user_move = int(input('Play Your Next Move: '))
  while game_state[user_move-1] != 0:
    user_move = int(input('That square is taken, play another move: '))
  # Make the player's move
  game_state[user_move-1] = 1

  # Print the new board after player move
  print_game(game_state)

  # Check for a tie
  if not 0 in game_state:
    print('It\'s a tie!')
    break
  # Check for Winner
  if check_win(game_state):
    # Game is over
    break

  # AI MOVE HANDLING
  ai_move_matrix = mlp_clf_optimal.predict(game_state.reshape(1, -1))
  ai_move_matrix = ai_move_matrix.flatten()
  # print(ai_move_matrix)
  # AI move is first optimal move in matrix
  for i in range(ai_move_matrix.size):
    if ai_move_matrix[i] == 1:
      game_state[i] = -1
      break

  # Print the new board after AI move
  print("The computer made a move: ")
  print_game(game_state)

  # Check for a tie
  if not 0 in game_state:
    print('It\'s a tie!')
    break
  # Check for Winner
  if check_win(game_state):
    # Game is over
    break
