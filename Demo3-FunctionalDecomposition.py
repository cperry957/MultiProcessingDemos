import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

outputDirectory = os.path.join("Output", "Demo3-FunctionalDecomposition")
outputFileName = "results.csv"

outputPath = os.path.join(outputDirectory, outputFileName)

if(os.path.exists(outputPath)):
    os.remove(outputPath)
else:
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

cmd = "g++-8 Demo3-FunctionalDecomposition.cpp -o Debug/prog -lm -fopenmp"
os.system( cmd )
cmd = "./Debug/prog >> " + outputPath
os.system( cmd )
df = pd.DataFrame(pd.read_csv(outputPath, names=['Month', 'DateMonth', 'DateYear', 'Temp', 'Precip', 'Height', 'NumDeer', 'NumCoyotes'], header=None))
df = df.assign(Temp_C = lambda x: (5/9) * (x.Temp - 32))
df = df.assign(Precip_cm = lambda x: (x.Precip * 2.54))
df = df.assign(Height_cm = lambda x: (x.Height * 2.54))
print(df)


#Create Imperial Chart
cm = plt.get_cmap('gist_earth')
fig, ax = plt.subplots()
df.plot(kind='line',x='Month',y='Temp',ax=ax, label='Temp (F)')
df.plot(kind='line',x='Month',y='Precip',ax=ax, label='Precip (in)')
df.plot(kind='line',x='Month',y='Height',ax=ax, label='Height (in)')
df.plot(kind='line',x='Month',y='NumDeer',ax=ax, label='Deer')
df.plot(kind='line',x='Month',y='NumCoyotes',ax=ax, label='Coyotes')
plt.title("Growth Simulation")
plt.xlabel("Months")
plt.legend(loc='upper left')
plt.savefig(os.path.join(outputDirectory, 'GrowthSimulation-Imperial.png'))

#Create Metric Chart
cm = plt.get_cmap('gist_earth')
fig, ax = plt.subplots()
df.plot(kind='line',x='Month',y='Temp_C',ax=ax, label='Temp (C)')
df.plot(kind='line',x='Month',y='Precip_cm',ax=ax, label='Precip (cm)')
df.plot(kind='line',x='Month',y='Height_cm',ax=ax, label='Height (cm)')
df.plot(kind='line',x='Month',y='NumDeer',ax=ax, label='Deer')
df.plot(kind='line',x='Month',y='NumCoyotes',ax=ax, label='Coyotes')
plt.title("Growth Simulation")
plt.xlabel("Months")
plt.legend(loc='upper left')
plt.savefig(os.path.join(outputDirectory, 'GrowthSimulation-Metric.png'))