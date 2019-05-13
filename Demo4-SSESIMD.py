import os
import csv
#import pandas as pd
#import matplotlib.pyplot as plt
from multiprocessing import cpu_count

outputDirectory = os.path.join("Output", "Demo4-SSESIMD")
outputFileName = "results.csv"
outputPath = os.path.join(outputDirectory, outputFileName)
sizes = [ 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216 ]

if(os.path.exists(outputPath)):
    os.remove(outputPath)
else:
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

for size in sizes:
    cmd = "g++ -c simd.p4.cpp -o Debug/simd.p4.o"
    os.system(cmd)
    cmd = "g++ -DARRAYSIZE=%d -o Debug/prog Demo4-SSESIMD.cpp Debug/simd.p4.o -lm -fopenmp" % ( size )
    os.system(cmd)
    cmd = "./Debug/prog >> " + outputPath
    os.system(cmd)

#df = pd.DataFrame(pd.read_csv(outputPath, names=['Type', 'ArraySize', 'MegaCalcsPerSecond', 'Sum'], header=None))

#print("Performance:")
#grouped_df = df.groupby(['ArraySize', 'Type'])['MegaCalcsPerSecond'].max()
#report_df = pd.DataFrame(grouped_df.to_frame().unstack().MegaCalcsPerSecond)

#report_df = report_df.assign(Speed_Up_Mul = lambda x: x.SimdMul / x.NonSimdMul)
#report_df = report_df.assign(Speed_Up_MulSum = lambda x: x.SimdMulSum / x.NonSimdMulSum)
#print(report_df)

##Create Speed Up Chart
#cm = plt.get_cmap('gist_earth')
#fig, ax = plt.subplots()
#report_df.plot(kind='line',y='Speed_Up_Mul',ax=ax, label='Speed Up - Mul')
#report_df.plot(kind='line',y='Speed_Up_MulSum',ax=ax, label='Speed Up - MulSum')
#ax.set_xscale('log', basex=2)
#plt.title("Speed Ups")
#plt.xlabel("Log_2(Array Size)")
#plt.ylabel("Speed Up")
#plt.legend(loc='upper left')
#plt.savefig(os.path.join(outputDirectory, 'SpeedUp.png'))

##Create Performance Chart
#cm = plt.get_cmap('gist_earth')
#fig, ax = plt.subplots()
#report_df.plot(kind='line',y='SimdMulSum',ax=ax, label='Simd - Mul Sum')
#report_df.plot(kind='line',y='SimdMul',ax=ax, label='Simd - Mul')
#report_df.plot(kind='line',y='NonSimdMulSum',ax=ax, label='Non-Simd - Mul Sum')
#report_df.plot(kind='line',y='NonSimdMul',ax=ax, label='Non-Simd - Mul')

#ax.set_xscale('log', basex=2)
#plt.title("Performance")
#plt.xlabel("Log_2(Array Size)")
#plt.ylabel("MegaCalcs/Second")
#plt.legend(loc='upper left')
#plt.savefig(os.path.join(outputDirectory, 'Performance.png'))
