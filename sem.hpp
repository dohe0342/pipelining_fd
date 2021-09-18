//
//  sem.hpp
//  pipelining_ex
//
//  Created by 김도희 on 2020/02/25.
//  Copyright © 2020 김도희. All rights reserved.
//

#ifndef SEM_H
#define SEM_H 1

#include <pthread.h>

typedef struct sem {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int count;
} sem_t;

extern void sem_init( sem_t *s, int init );
extern void sem_p( sem_t *s, int pid, const char *name );
extern void sem_v( sem_t *s, int pid, const char *name );

#endif
     
