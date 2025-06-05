def display_Path_to_Princess(N, grid):
    #To make sure that the bot is always at the centre of the grid
    bot_r, bot_c = N // 2, N // 2
    #To Find princess location (only at one of the corners)
    for i in [0, N - 1]:
        for j in [0, N - 1]:
            if grid[i][j] == 'p':
                princess_r, princess_c = i, j
    row_diff = princess_r - bot_r
    if row_diff < 0:
        print('UP\n' * abs(row_diff), end='')
    else:
        print('DOWN\n' * row_diff, end='')
    col_diff = princess_c - bot_c
    if col_diff < 0:
        print('LEFT\n' * abs(col_diff), end='')
    else:
        print('RIGHT\n' * col_diff, end='')

if __name__ == '__main__':
    N = int(input())  #Reads the size of the grid
    grid = []
    for _ in range(N):
        grid.append(input().strip())  #Reads every row in the grid

    display_Path_to_Princess(N, grid)
