//
//  Header.h
//  pipelining_ex
//
//  Created by 김도희 on 2020/02/25.
//  Copyright © 2020 김도희. All rights reserved.
//

#define TASK_NUM    2
#define NUM_FRAME    10000

// Thread data structures
pthread_t pid[TASK_NUM];

sem_t join_sem;
sem_t buffer[TASK_NUM];
sem_t exec_token[TASK_NUM];

struct _thread_arg {
    int thread_id;
    int frame_num;
} thread_arg[TASK_NUM];
