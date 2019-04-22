import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

outputDirectory = os.path.join("Output", "Demo1-MonteCarlo")
outputFileName = "results.csv"
outputPath = os.path.join(outputDirectory, outputFileName)
trials = [ 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 ]
threads = [ 1, 2, 4, 6, 8 ]
runs = 1

if(not os.path.exists(outputPath) or input("Would you like to rerun the Monte Carlo simulation? (Y - Yes)") == "Y"):
    if(os.path.exists(outputPath)):
        os.remove(outputPath)
    else:
        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)

    for trial in trials:
        for thread in threads:
            print("Trials = %d Threads = %d" % (trial, thread))
            cmd = "g++-8 -DNUMTRIALS=%d -DNUMT=%d Demo1-MonteCarlo.cpp -o Debug/prog -lm -fopenmp" % ( trial, thread )
            os.system( cmd )
            for run in range(runs):
                cmd = "./Debug/prog >> " + outputPath
                os.system( cmd )

df = pd.DataFrame(pd.read_csv(outputPath, names=['Threads', 'Trials', 'Probability', 'Performance'], header=None))
print("Probabilities:")
df_prob = df.groupby(['Trials'])["Probability"]
print(df_prob.describe())

print("Performance:")
print(df.groupby(['Threads', 'Trials'])['Performance'].describe())

report_df = df.loc[df.groupby(['Threads', 'Trials'])['Performance'].idxmax()]
single_df = report_df[report_df['Threads'] == report_df['Threads'].min()][['Trials', 'Performance']]
single_df.rename(columns={'Performance': 'T1_Performance'}, inplace=True)

report_df = pd.merge(report_df, single_df, left_on=['Trials'], right_on=['Trials'])
report_df = report_df.assign(Speed_Up = lambda x: x.Performance / x.T1_Performance)
report_df = report_df.assign(Parallel_Fraction = lambda x: (x.Threads/(x.Threads - 1)*(1-(1/x.Speed_Up))))
#report_df['Parallel_Fraction'] = report_df['Threads'] / (report_df['Threads'] - 1)
print(report_df)


#Create Performance vs. Number of Threads Chart
cm = plt.get_cmap('gist_earth')
fig, ax = plt.subplots()
ax.set_prop_cycle(color=[cm(1.*i/len(trials)) for i in range(len(trials))])
report_df.set_index('Threads', inplace=True)
report_df.groupby('Trials')['Performance'].plot(legend=True, marker=".")
plt.title("Performance vs. Number of Threads")
plt.legend(title="Trial Size")
plt.ylabel("MegaTrials Per Second")
plt.xlabel("Number of Threads")
plt.savefig(os.path.join(outputDirectory, 'PerfVsThreads.png'))

#Create Performance vs. Number of Trials Chart
plt.figure()
fig, ax = plt.subplots()
ax.set_prop_cycle(color=[cm(1.*i/len(threads)) for i in range(len(threads))])
report_df.reset_index(inplace=True)
report_df.set_index('Trials', inplace=True)
report_df.groupby('Threads')['Performance'].plot(legend=True, marker=".")
plt.title("Performance vs. Number of Trials")
plt.legend(title="Threads")
plt.ylabel("MegaTrials Per Second")
plt.xlabel("Number of Trials")
plt.savefig(os.path.join(outputDirectory, 'PerfVsTrials.png'))

#Create Performance vs. Number of Trials Chart - Log Scale
plt.figure()
fig, ax = plt.subplots()
ax.set_prop_cycle(color=[cm(1.*i/len(threads)) for i in range(len(threads))])
ax.set_xscale('log', basex=2)
report_df.reset_index(inplace=True)
report_df.set_index('Trials', inplace=True)
report_df.groupby('Threads')['Performance'].plot(legend=True, marker=".")
plt.title("Performance vs. Log(Number of Trials)")
plt.legend(title="Threads")
plt.ylabel("MegaTrials Per Second")
plt.xlabel("Log(Number of Trials)")
plt.savefig(os.path.join(outputDirectory, 'PerfVsTrialsLog.png'))