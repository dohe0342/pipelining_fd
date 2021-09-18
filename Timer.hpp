//
//  Timer.hpp
//  pipelining_ex
//
//  Created by 김도희 on 2020/02/25.
//  Copyright © 2020 김도희. All rights reserved.
//

//
//  Timer.h
//
//  Created by hylo on 4/22/14.
//  Modified by hyeonjin on 4/18/19.
//  Copyright (c) 2014 Parallel Software Design Lab, UOS. All rights reserved.
//

#ifndef HYLO_TIMER_H
#define HYLO_TIMER_H

#include <stdio.h>

const int MAX_NUM_TIMER = 100;
const int MAX_LEN_ALIAS = 100;

#ifndef DEBUG_PRINT
#define DEBUG_PRINT false
#endif

bool    initTimer();
void    startTimer(const char* timerName );
double    endTimer(const char* timerName );
double    endTimerp(const char* timerName );
double    getTimer(const char* timerName );
void    setTimer(const char* timerName );
void    printTimer(const char* timerName );
void    printTimerAcc(const char* timerName );
void    printTimerAvg(const char* timerName);
void    destroyTimer();

#if (DEBUG_PRINT == true)
    #define D_Start_Timer(str)     startTimer(str)
    #define D_End_Timer(str)       endTimer(str)
    #define D_End_Timerp(str)      endTimerp(str)
    #define D_Get_Timer(str)       getTimer(str)
    #define D_Set_Timer(str)       setTimer(str)
    #define D_Print_Timer(str)     printTimer(str)
    #define D_Print_Timer_Acc(str) printTimerAcc(str)
    #define D_Print_Timer_Avc(str) printTimerAvc(str)
#else
    #define D_Start_Timer(str)
    #define D_End_Timer(str)
    #define D_End_Timerp(str)
    #define D_Get_Timer(str)
    #define D_Set_Timer(str)
    #define D_Print_Timer(str)
    #define D_Print_Timer_Acc(str)
    #define D_Print_Timer_Avc(str)
#endif

#define Init_Timer()         initTimer()
#define Start_Timer(str)     startTimer(str)
#define End_Timer(str)       endTimer(str)
#define End_Timerp(str)      endTimerp(str)
#define Get_Timer(str)       getTimer(str)
#define Set_Timer(str)       setTimer(str)
#define Print_Timer(str)     printTimer(str)
#define Print_Timer_Acc(str) printTimerAcc(str)
#define Print_Timer_Avc(str) printTimerAvc(str)
#define Destroy_Timer()      destroyTimer()

#endif
