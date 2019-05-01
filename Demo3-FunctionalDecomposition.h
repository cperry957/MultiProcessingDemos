#pragma once

// function prototypes:
float Ranf(unsigned int*, float, float);
int Ranf(unsigned int*, int, int);
void TimeOfDaySeed();
void RunThread(float (*compute)(), void (*assign)(float), void (*print)());
float ComputeCoyotes();
void AssignCoyotes(float);
float ComputeGrainDeer();
void AssignGrainDeer(float);
float ComputeGrainGrowth();
void AssignGrainGrowth(float);
void ComputeEnvironment();
void FinishWatcher();
void InitBarrier(int);
void WaitBarrier();
void TimeOfDaySeed();
float SQR(float x) { return x * x; }