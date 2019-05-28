import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from multiprocessing import cpu_count

outputDirectory = os.path.join("Output", "Demo5-OpenCLArrays")
outputFileName = "results.csv"
outputPath = os.path.join(outputDirectory, outputFileName)
sizes = [ 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608 ]
local_sizes = [ 8, 16, 32, 64, 128, 256, 512, 1024 ]

titlePrefix = "Rabbit - "

#if(os.path.exists(outputPath)):
#    os.remove(outputPath)
#else:
#    if not os.path.exists(outputDirectory):
#        os.makedirs(outputDirectory)

#for size in sizes:
#    for local_size in local_sizes:
#        cmd = "g++ -DNUM_ELEMENTS=%d -DLOCAL_SIZE=%d -o Debug/Demo5-OpenCLArrays Demo5-OpenCLArrays.cpp /usr/lib/x86_64-linux-gnu/libOpenCL.so -lm -fopenmp" % ( size, local_size )
#        os.system(cmd)
#        cmd = "./Debug/Demo5-OpenCLArrays >> " + outputPath
#        os.system(cmd)

df = pd.DataFrame(pd.read_csv(outputPath, names=['Function', 'Size', 'Local_Size', 'Number_Work_Groups', 'GigaCalcsPerSecond'], header=None))
df = df[df.Size <= 8388608]
print("Performance:")
print(df)

#Create Performance Vs. Global Work-Size Chart
report_df = pd.DataFrame(df[(df.Function == "ArrayMulti") | (df.Function == "ArrayMultiAdd")][['Function', 'Size', 'Local_Size', 'GigaCalcsPerSecond']]).pivot_table(index='Size', columns=['Function','Local_Size'], values='GigaCalcsPerSecond')
print(report_df)
report_df.to_excel(os.path.join(outputDirectory, titlePrefix + "PrefVsGlobal.xlsx"))

fig = plt.figure(figsize=(16,9), dpi=300)
ax = fig.add_subplot(111)
multNewLabels = list(["ArrayMulti - " + str(x) for x in np.array(report_df.ArrayMulti.axes[1].tolist())])
addNewLabels = list(["ArrayMultiAdd - " + str(x) for x in np.array(report_df.ArrayMultiAdd.axes[1].tolist())])
newLabels = multNewLabels + addNewLabels

#Array Mult & Sum
cm = plt.get_cmap('Reds')
ax.set_prop_cycle(color=[cm((.9*i/len(multNewLabels))+.1) for i in range(len(multNewLabels))])
report_df.ArrayMulti.plot(kind='line', ax=ax)
cm = plt.get_cmap('Blues')
ax.set_prop_cycle(color=[cm((.9*i/len(multNewLabels))+.1) for i in range(len(addNewLabels))])
report_df.ArrayMultiAdd.plot(kind='line', ax=ax)

ax.set_xscale('log', basex=2)
#ax.xaxis.set_ticks(report_df.ArrayMultiAdd.axes[0].tolist())
#ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
#ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())

handles, labels = ax.get_legend_handles_labels()
plt.subplots_adjust(top=0.8)
plt.legend(handles, newLabels, prop={'size': 8}, bbox_to_anchor=(0., 1.05, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
plt.title(titlePrefix + "Giga Operations/Second Vs. Global Work Size")
plt.xlabel("Log_2(Global Work Sizes)")
plt.ylabel("Giga Operations/Second")
ax.set_ylim(bottom=0)
plt.savefig(os.path.join(outputDirectory, titlePrefix + 'PerfVsGlobal.png'))

#Create Performance Vs. Local Work-Size Chart
report_df = pd.DataFrame(df[(df.Function == "ArrayMulti") | (df.Function == "ArrayMultiAdd")][['Function', 'Size', 'Local_Size', 'GigaCalcsPerSecond']]).pivot_table(index='Local_Size', columns=['Function','Size'], values='GigaCalcsPerSecond')
print(report_df)
report_df.to_excel(os.path.join(outputDirectory, titlePrefix + "PerfVsLocal.xlsx"))

fig = plt.figure(figsize=(16,9), dpi=300)
ax = fig.add_subplot(111)
multNewLabels = list(["ArrayMulti - " + str(x) for x in np.array(report_df.ArrayMulti.axes[1].tolist())])
addNewLabels = list(["ArrayMultiAdd - " + str(x) for x in np.array(report_df.ArrayMultiAdd.axes[1].tolist())])
newLabels = multNewLabels + addNewLabels

#Array Mult & Sum
cm = plt.get_cmap('Reds')
ax.set_prop_cycle(color=[cm((.9*i/len(multNewLabels))+.1) for i in range(len(multNewLabels))])
report_df.ArrayMulti.plot(kind='line', ax=ax)

cm = plt.get_cmap('Blues')
ax.set_prop_cycle(color=[cm((.9*i/len(multNewLabels))+.1) for i in range(len(addNewLabels))])
report_df.ArrayMultiAdd.plot(kind='line', ax=ax)

ax.set_xscale('log', basex=2)
ax.xaxis.set_ticks([8, 16, 32, 64, 128, 256, 512, 1024])
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

handles, labels = ax.get_legend_handles_labels()
plt.subplots_adjust(top=0.7)
plt.legend(handles, newLabels, prop={'size': 8}, bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
plt.title(titlePrefix + "Giga Operations/Second Vs. Local Work Group Size")
plt.xlabel("Local Work Sizes")
plt.ylabel("Giga Operations/Second")
ax.set_ylim(bottom=0)
plt.savefig(os.path.join(outputDirectory, titlePrefix + 'PerfVsLocal.png'))

#Create Performance Vs. Global Work-Size Chart
report_df = pd.DataFrame(df[(df.Function == "ArrayMultiReduce") & (df.Local_Size >= 32) & (df.Local_Size <= 256)][['Function', 'Size', 'Local_Size', 'GigaCalcsPerSecond']]).pivot_table(index='Size', columns=['Function','Local_Size'], values='GigaCalcsPerSecond')
print(report_df)
report_df.to_excel(os.path.join(outputDirectory, titlePrefix + "ReductionPerfVsGlobal.xlsx"))

fig = plt.figure(figsize=(16,9), dpi=300)
ax = fig.add_subplot(111)
newLabels = list(["ArrayMultiReduce - " + str(x) for x in np.array(report_df.ArrayMultiReduce.axes[1].tolist())])

#Array Mult Reduction
cm = plt.get_cmap('gist_earth')
ax.set_prop_cycle(color=[cm((.9*i/len(newLabels))+.1) for i in range(len(newLabels))])
report_df.ArrayMultiReduce.plot(kind='line', ax=ax)

ax.set_xscale('log', basex=2)
ax.xaxis.set_ticks(report_df.ArrayMultiReduce.axes[0].tolist())
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

handles, labels = ax.get_legend_handles_labels()
plt.subplots_adjust(top=0.8)
plt.legend(handles, newLabels, prop={'size': 8}, bbox_to_anchor=(0., 1.05, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
plt.title(titlePrefix + "Giga Reductions/Second Vs. Global Work Size")
plt.xlabel("Log_2(Global Work Sizes)")
plt.ylabel("Giga Reductions/Second")
ax.set_ylim(bottom=0)
plt.savefig(os.path.join(outputDirectory, titlePrefix + 'ReductionPerfVsGlobal.png'))