//
//  Timer.cpp
//  pipelining_ex
//
//  Created by 김도희 on 2020/02/25.
//  Copyright © 2020 김도희. All rights reserved.
//
//
//  Timer.cpp
//
//  Created by hylo on 4/22/14.
//  Copyright (c) 2014 Parallel Software Design Lab, UOS. All rights reserved.
//

#include "Timer.hpp"
#include "string.h"

#ifdef _WIN32
#include <Windows.h>
LARGE_INTEGER li_st[MAX_NUM_TIMER];
LARGE_INTEGER li_ed[MAX_NUM_TIMER];
double PCFreq = 0.0;
#else
#include <sys/time.h>
struct timeval tv_st[MAX_NUM_TIMER],
                tv_ed[MAX_NUM_TIMER], tv_diff[MAX_NUM_TIMER];
#endif

double t[MAX_NUM_TIMER] = {0,};
double prev_t[MAX_NUM_TIMER] = {0,};
double count[MAX_NUM_TIMER] = {0,};
int g_num_timer = 0;
const char* timer_alias[MAX_NUM_TIMER];


// Internal function that is called inside the external APIs
inline int getTimerIndex(const char* timerName )
{
    // Check already exist timer
    int i;
    for( i = 0 ; i < g_num_timer; ++i ) {
        if( !strcmp(timer_alias[i], timerName) )
            break;
    }

    return i;
}

bool initTimer()
{
    g_num_timer = 0;
#ifdef _WIN32
    if(!QueryPerformanceFrequency(&li_st[0])) {
        printf("QueryPerformanceFrequency Failed!\n");
        return false;
    }
    PCFreq = (double)li_st[0].QuadPart/1000.0;
#else

#endif

    return true;
}


void startTimer(const char* timerName )
{
    int i = getTimerIndex(timerName);

    if( g_num_timer >= MAX_NUM_TIMER) {
        printf("Exceeded maximum number of timers\n");
        return;

    } else if( i == g_num_timer ) {
        timer_alias[g_num_timer] = timerName;
        g_num_timer++;
    }
    int target = i;

    count[target]++;
#ifdef _WIN32
    QueryPerformanceCounter(&li_st[target]);
#else
    gettimeofday(&tv_st[target], NULL);
#endif

}


double endTimer(const char* timerName )
{
    // Find exist timer
    int i = getTimerIndex(timerName);

    if( i == g_num_timer ) {
        printf("error: cannot find the timer \"%s\"\n", timerName);
        return -1;
    }

    int target = i;
    prev_t[target] = t[target];
#ifdef _WIN32
    QueryPerformanceCounter(&li_ed[target]);
    t[target] += (double)(li_ed[target].QuadPart - li_st[target].QuadPart)/PCFreq;
#else
    gettimeofday(&tv_ed[target], NULL);
    timersub(&tv_ed[target], &tv_st[target], &tv_diff[target]);
    t[target] += tv_diff[target].tv_sec * 1000.0 + tv_diff[target].tv_usec/1000.0;
#endif

    return t[target];
}


double endTimerp(const char* timerName )
{
    // Find exist timer
    int i = getTimerIndex(timerName);

    if( i == g_num_timer ) {
        printf("error: cannot find the timer \"%s\"\n", timerName);
        return -1;
    }

    int target = i;
    prev_t[target] = t[target];
#ifdef _WIN32
    QueryPerformanceCounter(&li_ed[target]);
    t[target] += (double)(li_ed[target].QuadPart - li_st[target].QuadPart)/PCFreq;
#else
    gettimeofday(&tv_ed[target], NULL);
    timersub(&tv_ed[target], &tv_st[target], &tv_diff[target]);
    t[target] += tv_diff[target].tv_sec * 1000.0 + tv_diff[target].tv_usec/1000.0;
#endif

    printf("[Timer] %s: %.2lf ms\n", timerName, t[target] - prev_t[target]);
    return t[target];
}

double getTimer(const char* timerName )
{
    // Find exist timer
    int i = getTimerIndex(timerName);

    if( i == g_num_timer ) {
        printf("error: cannot find the timer \"%s\"\n", timerName);
        return -1;
    }

    int target = i;
    return t[target];
}


void setTimer(const char* timerName )
{
    int i = getTimerIndex(timerName);

    if( i == g_num_timer ) {
        printf("error: cannot find the timer \"%s\"\n", timerName);
        return;
    }

    int target = i;
    t[target] = 0;
    prev_t[target] = 0;
    count[target] = 0;
}


void printTimer(const char* timerName )
{
    int i = getTimerIndex(timerName);

    if( i == g_num_timer ) {
        printf("error: cannot find the timer \"%s\"\n", timerName);
        return;
    }

    int target = i;

    printf("[Timer] %s: %.2lf ms\n", timerName, t[target] - prev_t[target]);
}


void printTimerAcc(const char* timerName ) {
    
    int i = getTimerIndex(timerName);

    if( i == g_num_timer ) {
        printf("error: cannot find the timer \"%s\"\n", timerName);
        return;
    }

    int target = i;

    printf("[Timer] %s: %.2lf ms\n", timerName, t[target]);
}


void printTimerAvg(const char* timerName)
{
    int i = getTimerIndex(timerName);

    if( i == g_num_timer ) {
        printf("error: cannot find the timer \"%s\"\n", timerName);
        return;
    }

    int target = i;
    printf("[Timer] %s: %.2lf ms\n", timerName, t[target]/count[target]);
}
