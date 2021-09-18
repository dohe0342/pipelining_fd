//
//  sem.cpp
//  pipelining_ex
//
//  Created by 김도희 on 2020/02/25.
//  Copyright © 2020 김도희. All rights reserved.
//

#include "sem.hpp"
#include <stdio.h>

//#define DEBUG

void sem_init( sem_t *s, int init ) {
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->cond, NULL);
        s->count = (init < 0 ? 0 : init);
#ifdef DEBUG
        printf("entering sem_init. The count value = %d, %u\n", s->count,  s);
#endif
}

void sem_p( sem_t *s, int tid, const char *name ) {
#ifdef DEBUG
        //printf("tid %d is entering sem_p. The count value = %d, %u\n", tid, s->count, s);
        printf("tid %d is entering sem_p to acquire %s. The count value = %d, %u\n", tid, name, s->count, s);
#endif
        pthread_mutex_lock(&s->lock);
        while( s->count==0 )
                pthread_cond_wait(&s->cond, &s->lock);
        s->count--;
        pthread_mutex_unlock(&s->lock);
}

int sem_test( sem_t *s) {
        int test;
        pthread_mutex_lock(&s->lock);
        test=s->count;
        pthread_mutex_unlock(&s->lock);
        return test;
}

void sem_v( sem_t *s, int tid, const char *name ) {
        int available;

        pthread_mutex_lock(&s->lock);
        available = (s->count++ == 0);
        if( available ) {
                pthread_cond_signal(&s->cond);
        }
#ifdef DEBUG
        //printf("tid %d is entering sem_v. The count value = %d, %u\n", tid, s->count,  s);
        printf("tid %d is entering sem_v to release %s. The count value = %d, %u\n", tid, name, s->count,  s);
#endif
        pthread_mutex_unlock(&s->lock);
}

