#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include "Demo3-FunctionalDecomposition.h"

#ifndef INCLUDE_COYOTES
#define INCLUDE_COYOTES	1
#endif // !INCLUDE_COYOTES


omp_lock_t	Lock;
int		NumInThreadTeam;
int		NumAtBarrier;
int		NumGone;

int	NowYear;		// 2019 - 2024
int	NowMonth;		// 0 - 11
int PrintMonth;

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	NowNumDeer;		// number of deer in the current population
int NowNumCoyotes;	// number of coyotes in the current population

unsigned int seed = 0;  // a thread-private variable
float x = Ranf(&seed, -1.f, 1.f);

const float GRAIN_GROWS_PER_MONTH = 8.0;
const float ONE_DEER_EATS_PER_MONTH = 0.5;
const float ONE_COYOTES_EATS_PER_MONTH = 0.25;
const int COYOTE_FIELD_ATTRACTIVENESS = 5;

const float AVG_PRECIP_PER_MONTH = 6.0;	// average
const float AMP_PRECIP_PER_MONTH = 6.0;	// plus or minus
const float RANDOM_PRECIP = 2.0;	// plus or minus noise

const float AVG_TEMP = 50.0;	// average
const float AMP_TEMP = 20.0;	// plus or minus
const float RANDOM_TEMP = 10.0;	// plus or minus noise

const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;

const short LAST_YEAR = 2025;
const short LAST_MONTH = 12;
const short FIRST_YEAR = 2019;
const short FIRST_MONTH = 0;

//Units of grain growth are inches.
//Units of temperature are degrees Fahrenheit(°F).
//Units of precipitation are inches.

int main() {

	//Initialize
	TimeOfDaySeed();
	NowYear = FIRST_YEAR;
	NowMonth = FIRST_MONTH;
	PrintMonth = 1;
	
	ComputeEnvironment();
	
	omp_set_num_threads(4);	// same as # of sections
	InitBarrier(4);
#pragma omp parallel sections
	{
#pragma omp section
		{
			//Grain Deer
			RunThread(ComputeGrainDeer, AssignGrainDeer, NULL);
		}

#pragma omp section
		{
			//Grain
			RunThread(ComputeGrainGrowth, AssignGrainGrowth, NULL);
		}

#pragma omp section
		{
			//MyAgent
			RunThread(ComputeCoyotes, AssignCoyotes, NULL);
		}

#pragma omp section
		{
			//Watcher
			RunThread(NULL, NULL, FinishWatcher);
		}
	}       // implied barrier -- all functions must return in order
		// to allow any of them to get past here
}

void RunThread(float (*compute)(), void (*assign)(float), void (*print)())
{
	while (NowYear < LAST_YEAR)
	{
		// compute a temporary next-value for this quantity
		// based on the current state of the simulation:
		float computeValue = 0;;
		if (compute != 0) {
			computeValue = compute();
		}

		// DoneComputing barrier:
		WaitBarrier();
		if (assign != 0) {
			assign(computeValue);
		}

		// DoneAssigning barrier:
		WaitBarrier();

		if (print != 0) {
			print();
		}
		// DonePrinting barrier:
		WaitBarrier();
	}
}

float ComputeCoyotes()
{
	if (NowNumDeer < COYOTE_FIELD_ATTRACTIVENESS)
		return (NowNumCoyotes - 2);
	else
		return (NowNumCoyotes + 1);
}

void AssignCoyotes(float value)
{
	NowNumCoyotes = value <= 0 ? 0 : value;
}

float ComputeGrainDeer()
{
	int calcDeer = NowNumDeer - (NowNumCoyotes * ONE_COYOTES_EATS_PER_MONTH);
	if (NowNumDeer > NowHeight)
		calcDeer--;
	else
		calcDeer++;

	return calcDeer;
}

void AssignGrainDeer(float value)
{
	NowNumDeer = value <= 0 ? 0 : value;
}

float ComputeGrainGrowth()
{
	float tempFactor = exp(-SQR(((double)NowTemp - MIDTEMP) / 10.));
	float precipFactor = exp(-SQR(((double)NowPrecip - MIDPRECIP) / 10.));
	float calcHeight = NowHeight;
	calcHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
	calcHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;

	return calcHeight;
}

void AssignGrainGrowth(float value)
{
	NowHeight = value <= 0 ? 0 : value;
}

void ComputeEnvironment()
{
	float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);

	float temp = AVG_TEMP - AMP_TEMP * cos(ang);
	NowTemp = temp + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP);

	float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
	NowPrecip = precip + Ranf(&seed, -RANDOM_PRECIP, RANDOM_PRECIP);
	if (NowPrecip < 0.)
		NowPrecip = 0.;
}

void FinishWatcher()
{
	printf("%d,%d,%d,%lf,%lf,%lf,%d,%d\n", PrintMonth, (NowMonth + 1), NowYear, NowTemp, NowPrecip, NowHeight, NowNumDeer, NowNumCoyotes);
	
	if (++NowMonth == LAST_MONTH)
	{
		NowMonth = FIRST_MONTH;
		NowYear++;
	}
	PrintMonth++;
	ComputeEnvironment();
}

// specify how many threads will be in the barrier:
//	(also init's the Lock)

void InitBarrier(int n)
{
	NumInThreadTeam = n;
	NumAtBarrier = 0;
	omp_init_lock(&Lock);
}


// have the calling thread wait here until all the other threads catch up:

void WaitBarrier()
{
	omp_set_lock(&Lock);
	{
		NumAtBarrier++;
		if (NumAtBarrier == NumInThreadTeam)
		{
			NumGone = 0;
			NumAtBarrier = 0;
			// let all other threads get back to what they were doing
			// before this one unlocks, knowing that they might immediately
			// call WaitBarrier( ) again:
			while (NumGone != NumInThreadTeam - 1);
			omp_unset_lock(&Lock);
			return;
		}
	}
	omp_unset_lock(&Lock);

	while (NumAtBarrier != 0);	// this waits for the nth thread to arrive

#pragma omp atomic
	NumGone++;			// this flags how many threads have returned
}

float Ranf(unsigned int* seedp, float low, float high)
{
#ifdef _WIN32
	float r = (float)rand();              // 0 - RAND_MAX
#else
	float r = (float)rand_r(seedp);              // 0 - RAND_MAX
#endif

	return(low + r * (high - low) / (float)RAND_MAX);
}

int Ranf(unsigned int* seedp, int ilow, int ihigh)
{
	float low = (float)ilow;
	float high = (float)ihigh + 0.9999f;

	return (int)(Ranf(seedp, low, high));
}

void TimeOfDaySeed()
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time(&timer);
	double seconds = difftime(timer, mktime(&y2k));
	seed = (unsigned int)(1000. * seconds);    // milliseconds
}