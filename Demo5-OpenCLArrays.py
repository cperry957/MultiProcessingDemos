import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

outputDirectory = os.path.join("Output", "Demo5-OpenCLArrays")
outputFileName = "results.csv"
outputPath = os.path.join(outputDirectory, outputFileName)
sizes = [ 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608 ]
local_sizes = [ 8, 16, 32, 64, 128, 256, 512, 1024 ]

if(os.path.exists(outputPath)):
    os.remove(outputPath)
else:
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

for size in sizes:
    for local_size in local_sizes:
        cmd = "g++ -DNUM_ELEMENTS=%d -DLOCAL_SIZE=%d -o Debug/Demo5-OpenCLArrays Demo5-OpenCLArrays.cpp /usr/lib/x86_64-linux-gnu/libOpenCL.so -lm -fopenmp" % ( size, local_size )
        os.system(cmd)
        cmd = "./Debug/Demo5-OpenCLArrays >> " + outputPath
        os.system(cmd)

df = pd.DataFrame(pd.read_csv(outputPath, names=['Function', 'Size', 'Local_Size', 'Number_Work_Groups', 'MegaCalcsPerSecond'], header=None))

print("Performance:")
print(df)