import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

outputDirectory = os.path.join("Output", "Demo4-SSESIMD")
outputFileName = "results.csv"
outputPath = os.path.join(outputDirectory, outputFileName)
sizes = [ 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304 ]

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

#df = pd.DataFrame(pd.read_csv(outputPath, names=['Month', 'DateMonth', 'DateYear', 'Temp', 'Precip', 'Height', 'NumDeer', 'NumCoyotes'], header=None))
#df = df.assign(Temp_C = lambda x: (5/9) * (x.Temp - 32))
#df = df.assign(Precip_cm = lambda x: (x.Precip * 2.54))
#df = df.assign(Height_cm = lambda x: (x.Height * 2.54))
#print(df)


##Create Imperial Chart
#cm = plt.get_cmap('gist_earth')
#fig, ax = plt.subplots()
#df.plot(kind='line',x='Month',y='Temp',ax=ax, label='Temp (F)')
#df.plot(kind='line',x='Month',y='Precip',ax=ax, label='Precip (in)')
#df.plot(kind='line',x='Month',y='Height',ax=ax, label='Height (in)')
#df.plot(kind='line',x='Month',y='NumDeer',ax=ax, label='Deer')
#df.plot(kind='line',x='Month',y='NumCoyotes',ax=ax, label='Coyotes')
#plt.title("Growth Simulation - Imperial")
#plt.xlabel("Simulation Month")
#ax.set_xlim(left=0)
#plt.ylabel("Quantity (See Legend)")
#plt.legend(loc='upper left')
#plt.savefig(os.path.join(outputDirectory, 'GrowthSimulation-Imperial.png'))

##Create Metric Chart
#cm = plt.get_cmap('gist_earth')
#fig, ax = plt.subplots()
#df.plot(kind='line',x='Month',y='Temp_C',ax=ax, label='Temp (C)')
#df.plot(kind='line',x='Month',y='Precip_cm',ax=ax, label='Precip (cm)')
#df.plot(kind='line',x='Month',y='Height_cm',ax=ax, label='Height (cm)')
#df.plot(kind='line',x='Month',y='NumDeer',ax=ax, label='Deer')
#df.plot(kind='line',x='Month',y='NumCoyotes',ax=ax, label='Coyotes')
#plt.title("Growth Simulation - Metric")
#plt.xlabel("Simulation Month")
#ax.set_xlim(left=0)
#plt.ylabel("Quantity (See Legend)")
#plt.legend(loc='upper left')
#plt.savefig(os.path.join(outputDirectory, 'GrowthSimulation-Metric.png'))