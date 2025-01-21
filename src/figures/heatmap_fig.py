from src.tools.myplot import heatmap
from src.tools.csv_process import *
for i in range(1,4):
    mat = read_csv_to_matrix(f'csvs/{i}.csv')
    heatmap(mat, isShow=False, isSave=f'{i}.png')
