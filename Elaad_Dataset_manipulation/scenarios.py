# %% Libraries import
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from numpy import random
from scipy.stats import norm
import time
from utils import *
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()

# %% Options and data import

data_df = pd.read_excel('elaadnl_open_ev_datasets.xlsx', sheet_name='open_transactions')
startTime_col = 'UTCTransactionStart'
stopTime_col = 'UTCTransactionStop'
data_str = 'ElaadNL'

data_df['startTime'] = pd.to_datetime(data_df[startTime_col], utc=True)
data_df['startTime_local'] = data_df['startTime'].dt.tz_convert('Europe/Berlin')
data_df['stopTime'] = pd.to_datetime(data_df[stopTime_col], utc=True)
data_df['stopTime_local'] = data_df['stopTime'].dt.tz_convert('Europe/Berlin')
data_df['weekday'] = data_df['startTime_local'].dt.weekday

### Getting rid of weekend days and sessions that were not started and finished in the same day
# data_df = data_df[((data_df['weekday'] != 5) & (data_df['weekday'] != 6))]

#%% Sampling definition

sampling_freq = '1h'

# Ranges definition
time_range = (pd.date_range(pd.Timestamp('00:00:00'), pd.Timestamp('23:59:00'), freq=sampling_freq, inclusive='left')).time
step_power = 4
power_range = list(range(0, 12+step_power, step_power))
step_energy = 5
energy_range = list(range(0, 30+step_energy, step_energy))
connectedtime_range = list(range(0,11,1))
chargetime_range = list(range(0,11,1))

# Counting the position of the considered columns for the statistical analysis with respect to their respective sampling ranges
data_df['startTime_interval'] = np.searchsorted(time_range, data_df['startTime_local'].dt.time)-1
data_df['stopTime_interval'] = np.searchsorted(time_range, data_df['stopTime_local'].dt.time)-1
data_df['MaxPower_interval'] = np.searchsorted(power_range, data_df['MaxPower'])
data_df['TotalEnergy_interval'] = np.searchsorted(energy_range, data_df['TotalEnergy'])
data_df['ConnectedTime_interval'] = np.searchsorted(connectedtime_range, data_df['ConnectedTime'])
data_df['ChargeTime_interval'] = np.searchsorted(chargetime_range, data_df['ChargeTime'])


#%% MaxPower and TotalEnergy conditional dependencies

# TotalEnergy pdf evaluation
init = pd.Series(np.zeros(len(energy_range)+1))
TotalEnergy_count = (pd.concat([init, data_df['TotalEnergy_interval'].value_counts()], axis=1).reindex(init.index).fillna(0))['count']
TotalEnergy_pdf = pd.DataFrame({'energy_level':energy_range + [energy_range[-1] + 5], 'occurrence': TotalEnergy_count.values.astype(float),
                              'pdf': TotalEnergy_count.values/sum(TotalEnergy_count.values)})
TotalEnergy_pdf['cumulative_pdf'] = TotalEnergy_pdf['pdf'].cumsum()

# conditional probability for TotalEnergy given maxPower
TotalEnergy_cond = data_df.pivot_table(index='MaxPower_interval', columns='TotalEnergy_interval', aggfunc='size', fill_value=0)
count_values = ((TotalEnergy_cond<=5) & (TotalEnergy_cond>0)).sum().sum()
sum_values = TotalEnergy_cond[(TotalEnergy_cond>0) & (TotalEnergy_cond<=5)].sum().sum()
TotalEnergy_cond = TotalEnergy_cond.map(lambda x: replace_values(x, sum_values, count_values))


#%% startTime and StopTime conditional dependencies

# startTime pdf evaluation
init = pd.Series(np.zeros(len(time_range), dtype=int))
startTime_count = (pd.concat([init, data_df['startTime_interval'].value_counts()], axis=1).reindex(init.index).fillna(0))['count']
startTime_pdf = pd.DataFrame({'time_local':time_range[startTime_count.index], 'occurrence': startTime_count.values.astype(int),
                              'pdf': startTime_count.values/sum(startTime_count.values)})
startTime_pdf['cumulative_pdf'] = startTime_pdf['pdf'].cumsum()

# conditional probability for stopTime given startTime
pivot_stopTime = data_df.pivot_table(index='startTime_interval', columns='stopTime_interval', aggfunc='size', fill_value=0)
dailypivot_stopTime = pd.DataFrame(0.0, index=range(len(time_range)), columns=range(len(time_range)))
dailypivot_stopTime.iloc[pivot_stopTime.index, pivot_stopTime.columns.astype(int)] = pivot_stopTime

count_values = ((pivot_stopTime<=5) & (pivot_stopTime>0)).sum().sum()
sum_values = pivot_stopTime[(pivot_stopTime>0) & (pivot_stopTime<=5)].sum().sum()
df_replaced = pivot_stopTime.map(lambda x: replace_values(x, sum_values, count_values))
dailypivot_stopTime = pd.DataFrame(0.0, index=range(len(time_range)), columns=range(len(time_range)))
dailypivot_stopTime.iloc[df_replaced.index, df_replaced.columns.astype(int)] = df_replaced

sum_rows = dailypivot_stopTime.sum(axis=1)
stopTime_cond = dailypivot_stopTime.copy()
for index, somma in sum_rows.items():
    if somma != 0.0:
        stopTime_cond.loc[index] /= somma


# conditional probability for ChargedTime given ConnectedTime  
ChargeTime_cond = data_df.pivot_table(index='ConnectedTime_interval', columns='ChargeTime_interval', aggfunc='size', fill_value=0)
count_values = ((ChargeTime_cond<=5) & (ChargeTime_cond>0)).sum().sum()
sum_values = ChargeTime_cond[(ChargeTime_cond>0) & (ChargeTime_cond<=5)].sum().sum()
ChargeTime_cond = ChargeTime_cond.map(lambda x: replace_values(x, sum_values, count_values))

# conditional probability for ConnectedTime given ChargedTime
EnergyTime_cond = data_df.pivot_table(index='ConnectedTime_interval', columns='TotalEnergy_interval', aggfunc='size', fill_value=0)
count_values = ((EnergyTime_cond<=5) & (EnergyTime_cond>0)).sum().sum()
sum_values = EnergyTime_cond[(EnergyTime_cond>0) & (EnergyTime_cond<=5)].sum().sum()
EnergyTime_cond = EnergyTime_cond.map(lambda x: replace_values(x, sum_values, count_values))


# %% PLOTS Occurrences and Probability

path_plots = os.getcwd() + '/plots/'

savefig_flag=False
figsize=(10, 6)
fontsize=16
dpi=300

if sampling_freq == '15min':
    n_steps = 96
    n_branches = 24
    freq_label1 = 8
    freq_label2 = 16
elif sampling_freq == '30min':
    n_steps = 48
    n_branches = 12
    freq_label1 = 6
    freq_label2 = 8
elif sampling_freq == '1h':
    n_steps = 24
    n_branches = 5
    freq_label1 = 2
    freq_label2 = 6


### Heatmap for TotalEnergy conditional to PeakPower
power_bins = [str(power_range[i]) + '-' + str(power_range[i+1]) for i in range(len(power_range)-1)] + ['>' + str(power_range[-1])]
energy_bins = [str(energy_range[i]) + '-' + str(energy_range[i+1]) for i in range(len(energy_range)-1)] + ['>' + str(energy_range[-1])]

### Plot for startTime probability distribution
startTime_pdf['time_local_str'] = startTime_pdf['time_local'].apply(lambda x: x.strftime('%H:%M'))
orari_ogni_due_ore = startTime_pdf['time_local_str'][::freq_label1]

### Heatmap for ChargedTime conditional to ConnectedTime
connectedtime_bins = [str(connectedtime_range[i]) + '-' + str(connectedtime_range[i+1]) for i in range(len(connectedtime_range)-1)] + ['>' + str(connectedtime_range[-1])]
chargetime_bins = [str(chargetime_range[i]) + '-' + str(chargetime_range[i+1]) for i in range(len(chargetime_range)-1)] + ['>' + str(chargetime_range[-1])]


# %% SCENARIOS

multi_level_sampling = data_df.pivot_table(index=['MaxPower_interval','TotalEnergy_interval'], columns=['ConnectedTime_interval','ChargeTime_interval'], aggfunc='size', fill_value=0)

n_extractions= 10000
scenarios = []
scenarios_explained = []
while len(scenarios) < n_extractions:
    # print(k)
    startTime, x = roulette_wheel_startTime(startTime_pdf['cumulative_pdf'])
    stopTime, x = roulette_wheel_stopTime(startTime, stopTime_cond.apply(lambda row: row.cumsum(), axis=1))
    while startTime == stopTime:
        startTime, x = roulette_wheel_startTime(startTime_pdf['cumulative_pdf'])
        stopTime, x = roulette_wheel_stopTime(startTime, stopTime_cond.apply(lambda row: row.cumsum(), axis=1))    
    connected_time = (stopTime - startTime) % 24
    first_level =  multi_level_sampling.xs(np.searchsorted(connectedtime_range, connected_time), level='ConnectedTime_interval', axis=1, drop_level=False)
    power = random.choice(power_range[1:]+[power_range[-1] + step_power])
    second_level = first_level.xs(np.searchsorted(power_range, power), level='MaxPower_interval', drop_level=False)
    
    TotalEnergy_cond_filter = TotalEnergy_cond[TotalEnergy_cond.index == np.searchsorted(power_range, power)]
    sum_rows = TotalEnergy_cond_filter.sum(axis=1)
    TotalEnergy_cond_prob = TotalEnergy_cond_filter.copy()
    for index, somma in sum_rows.items():
        if somma != 0.0:
            TotalEnergy_cond_prob.loc[index] = TotalEnergy_cond_filter.loc[index].astype(float) / float(somma)
    TotalEnergy_cond_cumsum = TotalEnergy_cond_prob.apply(lambda row: row.cumsum(), axis=1)
    energy, x = roulette_wheel_TotalEnergy_cond(TotalEnergy_cond_cumsum)
    chargedEnergy = energy_range[energy-1]+step_energy/2

    third_level = second_level.xs(np.searchsorted(energy_range, chargedEnergy), level='TotalEnergy_interval', drop_level=False)
    sum_rows = third_level.sum(axis=1)
    third_level_prob = third_level.copy()
    for index, somma in sum_rows.items():
        if somma != 0.0:
            third_level_prob.loc[index] = third_level.loc[index].astype(float) / float(somma)
    third_level_cumsum = third_level_prob.apply(lambda row: row.cumsum(), axis=1)
    chargeTime, x = roulette_wheel_thirdlevel(third_level_cumsum)
    if not np.isnan(chargeTime):
        scenarios.append((startTime, stopTime, connected_time, 0.9*power, chargedEnergy, chargeTime))
        scenarios_explained.append((time_range[startTime], time_range[stopTime], connected_time, power_bins[np.searchsorted(power_range, power)-1], energy_bins[np.searchsorted(energy_range, chargedEnergy)-1], chargeTime))

scenarios_df = pd.DataFrame(scenarios, columns=['arrival_time', 'departure_time', 'connection_time', 'peak_power_kW', 'charged_energy_kWh', 'charge_time'])
scenarios_explained_df = pd.DataFrame(scenarios_explained, columns=['arrival_time', 'departure_time', 'connection_time', 'peak_power_kW', 'charged_energy_kWh', 'charge_time'])

# %% Scenarios plots

scenarios_df_plot = scenarios_df[scenarios_df['charge_time'].isna() == False]

all_y_values = np.zeros((24, len(scenarios_df_plot)))

fig, ax = plt.subplots(figsize=(12, 6))
for i, (index, row) in enumerate(scenarios_df_plot.iterrows()):
    arrival_time = row['arrival_time'].astype(int)
    departure_time = row['departure_time'].astype(int)
    peak_power = row['peak_power_kW']
    charged_energy = row['charged_energy_kWh']
    charge_time = row['charge_time'].astype(int)

    energy_hour = (charged_energy % peak_power) / (charge_time - (charged_energy // peak_power))
    power_values = []
    for _ in range(charge_time):
        if peak_power < charged_energy:
            power_value = peak_power
        else:
            power_value = energy_hour
        charged_energy -= power_value
        power_values.append(power_value)

    x_values = range(24)
    y_values = np.zeros(24)
    for j in range(charge_time):
        y_values[min((arrival_time + j), 24)-1] = power_values[j]
        if arrival_time + j >= 24:
            y_values[(arrival_time + j) % 24] = power_values[j]

    ax.stairs(y_values, label=f'Line {index+1}')
    all_y_values[:, i] = y_values

ax.set_xlim(0, 23)
ax.set_xticks(startTime_pdf.index, startTime_pdf['time_local'].astype(str).apply(lambda x: x[:5]), rotation=45, fontsize=fontsize)
ax.set_xlabel('Hour', fontsize=fontsize, fontweight='bold')
ax.set_ylabel('Power [kW]', fontsize=fontsize, fontweight='bold')
ax.set_title('Number of scenarios: ' + str(n_extractions), fontsize=fontsize, fontweight='bold')
plt.tight_layout()

# %% Fan chart percentiles

# percentiles = range(5,100,10)
# percentiles = range(0,110,10)
percentiles = [0,2,10,20,30,40,50,60,70,80,90,98,100]
significant_patterns = np.zeros((24, len(percentiles)))
for i in range(len(percentiles)):
    significant_patterns[:, i] = np.percentile(all_y_values, percentiles[i], axis=1)

columns_names = ['pct' + str(percentiles[i]/100) for i in range(len(percentiles))]
percentiles_df = pd.DataFrame(significant_patterns, columns=columns_names)
percentiles_df.insert(0, 'Time', time_range)

fig, ax = plt.subplots(figsize=(12, 6))
xs = np.arange(len(percentiles_df))
colors = plt.cm.Reds(np.linspace(0.15, 0.85, 6))
# for lower, upper, color in zip([f'pct0.{i}'+'5' for i in range(0, 5)], [f'pct0.{i}'+'5' for i in range(9, 4, -1)], colors):
lowers = columns_names[:6]
uppers = columns_names[7:][::-1]
for lower, upper, color in zip(lowers, uppers, colors):
    ax.fill_between(xs, percentiles_df[lower], percentiles_df[upper], color=color, label=lower + '-' + upper, step='post')
# ax.plot(xs, percentiles_df['pct0.5'], color='black', lw=2, label='Median')
# ax.plot(xs, np.mean(all_y_values, axis=1), color='black', lw=2, label='Mean', linestyle='--')
# ax.stairs(percentiles_df['pct0.5'], color='black', lw=2, label='Median')
ax.stairs(np.mean(all_y_values, axis=1), color='black', lw=2, label='Mean', linestyle='--')
ax.set_xticks(xs)
ax.set_xticklabels(percentiles_df['Time'].astype(str).apply(lambda x: x[:2]), rotation=45, fontsize=fontsize)
ax.legend(loc='upper left')
ax.margins(x=0)
ax.set_xlim(0,23)
ax.set_ylim(0, np.max(significant_patterns))
ax.set_xlabel('Hour', fontsize=fontsize, fontweight='bold')
ax.set_ylabel('Power [kW]', fontsize=fontsize, fontweight='bold')
ax.set_title('Number of scenarios: ' + str(n_extractions), fontsize=fontsize, fontweight='bold')
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
if savefig_flag:
    fig.savefig(path_plots + 'scenarios_fanchart.png', dpi=300)

if not savefig_flag:
    plt.show()
