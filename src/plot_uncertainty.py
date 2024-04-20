import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
import pandas as pd
import os 

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 8)
result_folder = 'purple-poisson-score-fgm-apg-scale-0.035-clip'
num_img = 20
niter = 17
################ read images ###################
pois_score = np.zeros([niter, num_img])
pg_score = np.zeros([niter, num_img])
pois_score_apg = np.zeros([niter, num_img])
pg_score_apg = np.zeros([niter, num_img])
# skipimg = [2, 18]
for i in range(num_img):
    # if i in skipimg:
    #     continue
    fullpath = os.path.join('../result', result_folder, str(i))
    pois_score_dir = os.path.join(fullpath, 'pois_score')
    pg_score_dir = os.path.join(fullpath, 'pg_score')
    pois_score_apg_dir = os.path.join(fullpath, 'pois_score_apg')
    pg_score_apg_dir = os.path.join(fullpath, 'pg_score_apg')
    pois_dir = os.path.join(fullpath, 'pois')
    with open(f'{pois_dir}/logfile.log', 'r') as f:
        last_line = f.readlines()[-1]
        nrmse = last_line.split(' ')[-1]
        pois_score[0, i] = nrmse 
        pg_score[0, i] = nrmse 
        pois_score_apg[0, i] = nrmse 
        pg_score_apg[0, i] = nrmse
    f.close()
    ########### pois score ############
    with open(f'{pois_score_dir}/logfile.log', 'r') as f:
        Lines = f.readlines()
        k = 1
        for line in Lines:
            if 'nrmse' in line:
                nrmse = line.split(' ')[-1]
                pois_score[k, i] = nrmse
                k += 1
        pois_score[k-1:, i] = pois_score[k-1, i]
    f.close()
    ############# pg score ############
    with open(f'{pg_score_dir}/logfile.log', 'r') as f:
        Lines = f.readlines()
        k = 1
        for line in Lines:
            if 'nrmse' in line:
                nrmse = line.split(' ')[-1]
                pg_score[k, i] = nrmse
                k += 1
        pg_score[k-1:, i] = pg_score[k-1, i]
    f.close()
    ############# pois score apg ############
    with open(f'{pois_score_apg_dir}/logfile.log', 'r') as f:
        Lines = f.readlines()
        k = 1
        for line in Lines:
            if 'nrmse' in line:
                nrmse = line.split(' ')[-1]
                pois_score_apg[k, i] = nrmse
                k += 1
        pois_score_apg[k-1:, i] = pois_score_apg[k-1, i]
    f.close()
    ############## pg score apg ##############
    with open(f'{pg_score_apg_dir}/logfile.log', 'r') as f:
        Lines = f.readlines()
        k = 1
        for line in Lines:
            if 'nrmse' in line:
                nrmse = line.split(' ')[-1]
                pg_score_apg[k, i] = nrmse
                k += 1
        pg_score_apg[k-1:, i] = pg_score_apg[k-1, i]
    f.close()

idx = 5
print(pois_score[:,idx])
print(pg_score[:,idx])
print(pois_score_apg[:,idx])
print(pg_score_apg[:,idx])

niter = 12
epochs = list(range(12))
pois_score_mean = np.nanmean(100 * pois_score, axis=1)[:niter]
pois_score_std = np.nanstd(100 * pois_score, axis=1)[:niter]
pg_score_mean = np.nanmean(100 * pg_score, axis=1)[:niter]
pg_score_std = np.nanstd(100 * pg_score, axis=1)[:niter]
pois_score_apg_mean = np.nanmean(100 * pois_score_apg, axis=1)[:niter]
pois_score_apg_std = np.nanstd(100 * pois_score_apg, axis=1)[:niter]
pg_score_apg_mean = np.nanmean(100 * pg_score_apg, axis=1)[:niter]
pg_score_apg_std = np.nanstd(100 * pg_score_apg, axis=1)[:niter]

with sns.axes_style("darkgrid"):
    ax.plot(epochs, pois_score_mean, label='Pois-WFS', linewidth=2, c=clrs[0])
    ax.fill_between(epochs, pois_score_mean-pois_score_std, pois_score_mean+pois_score_std ,
                    alpha=0.1, facecolor=clrs[0])
    ax.plot(epochs, pg_score_mean, label='PG-WFS', linewidth=2, c=clrs[1])
    ax.fill_between(epochs, pg_score_mean-pg_score_std, pg_score_mean+pg_score_std ,
                    alpha=0.1, facecolor=clrs[1])
    ax.plot(epochs, pois_score_apg_mean, label='Pois-AWFS', linewidth=2, c=clrs[4])
    ax.fill_between(epochs, pois_score_apg_mean-pois_score_apg_std, 
                    pois_score_apg_mean+pois_score_apg_std ,
                    alpha=0.1, facecolor=clrs[4])
    ax.plot(epochs, pg_score_apg_mean, label='PG-AWFS', linewidth=2, c=clrs[6])
    ax.fill_between(epochs, pg_score_apg_mean-pois_score_apg_std, 
                    pg_score_apg_mean+pois_score_apg_std ,
                    alpha=0.1, facecolor=clrs[6])
    ax.legend(prop={'family':'Times New Roman', 'size':20})
    plt.xlabel('niter', fontsize=20, **csfont)
    plt.ylabel('NRMSE (%)', fontsize=20, **csfont)
    # plt.ylim(0, 0.3)
    plt.savefig(f'../result/{result_folder}/uncertainty_test_{result_folder}.pdf')