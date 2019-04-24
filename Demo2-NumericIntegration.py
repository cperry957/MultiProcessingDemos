import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

outputDirectory = os.path.join("Output", "Demo2-NumericIntegration")
outputFileName = "results.csv"

outputPath = os.path.join(outputDirectory, outputFileName)
#nodes = [ 50, 100, 128, 150, 200, 256, 300, 400, 512, 750, 1000, 1024 ]
nodes = [ 128, 256, 512, 1024, 2048, 4096, 8192, 16384 ]
threads = [1, 2, 4, 6] + list(range(8, cpu_count()+1, 8))
runs = 1

if(not os.path.exists(outputPath) or input("Would you like to rerun the numeric integration calculations? (Y - Yes)") == "Y"):
    if(os.path.exists(outputPath)):
        os.remove(outputPath)
    else:
        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)

    for node in nodes:
        for thread in threads:
            print("Nodes = %d Threads = %d" % (node, thread))
            cmd = "g++-8 -DNUMNODES=%d -DNUMT=%d Demo2-NumericIntegration.cpp -o Debug/prog -lm -fopenmp" % ( node, thread )
            os.system( cmd )
            for run in range(runs):
                cmd = "./Debug/prog >> " + outputPath
                os.system( cmd )

df = pd.DataFrame(pd.read_csv(outputPath, names=['Threads', 'Nodes', 'Volume', 'Performance'], header=None))
print("Probabilities:")
df_prob = df.groupby(['Nodes'])["Volume"]
print(df_prob.describe())

print("Performance:")
print(df.groupby(['Threads', 'Nodes'])['Performance'].describe())

report_df = df.loc[df.groupby(['Threads', 'Nodes'])['Performance'].idxmax()]
single_df = report_df[report_df['Threads'] == report_df['Threads'].min()][['Nodes', 'Performance']]
single_df.rename(columns={'Performance': 'T1_Performance'}, inplace=True)

report_df = pd.merge(report_df, single_df, left_on=['Nodes'], right_on=['Nodes'])
report_df = report_df.assign(Speed_Up = lambda x: x.Performance / x.T1_Performance)
report_df = report_df.assign(Parallel_Fraction = lambda x: (x.Threads/(x.Threads - 1)*(1-(1/x.Speed_Up))))
#report_df['Parallel_Fraction'] = report_df['Threads'] / (report_df['Threads'] - 1)
print(report_df)


#Create Performance vs. Number of Threads Chart
cm = plt.get_cmap('gist_earth')
fig, ax = plt.subplots()
ax.set_prop_cycle(color=[cm(1.*i/len(nodes)) for i in range(len(nodes))])
report_df.set_index('Threads', inplace=True)
report_df.groupby('Nodes')['Performance'].plot(legend=True, marker=".")
plt.title("Performance vs. Number of Threads")
plt.legend(title="Trial Size")
plt.ylabel("MegaNodes Per Second")
plt.xlabel("Number of Threads")
plt.savefig(os.path.join(outputDirectory, 'PerfVsThreads.png'))

#Create Performance vs. Number of Nodes Chart
plt.figure()
fig, ax = plt.subplots()
ax.set_prop_cycle(color=[cm(1.*i/len(threads)) for i in range(len(threads))])
report_df.reset_index(inplace=True)
report_df.set_index('Nodes', inplace=True)
report_df.groupby('Threads')['Performance'].plot(legend=True, marker=".")
plt.title("Performance vs. Number of Nodes")
plt.legend(title="Threads")
plt.ylabel("MegaNodes Per Second")
plt.xlabel("Number of Nodes")
plt.savefig(os.path.join(outputDirectory, 'PerfVsNodes.png'))

#Create Performance vs. Number of Nodes Chart - Log Scale
plt.figure()
fig, ax = plt.subplots()
ax.set_prop_cycle(color=[cm(1.*i/len(threads)) for i in range(len(threads))])
ax.set_xscale('log', basex=2)
report_df.reset_index(inplace=True)
report_df.set_index('Nodes', inplace=True)
report_df.groupby('Threads')['Performance'].plot(legend=True, marker=".")
plt.title("Performance vs. Log(Number of Nodes)")
plt.legend(title="Threads")
plt.ylabel("MegaNodes Per Second")
plt.xlabel("Log(Number of Nodes)")
plt.savefig(os.path.join(outputDirectory, 'PerfVsNodesLog.png'))