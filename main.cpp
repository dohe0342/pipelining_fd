//
//  main.cpp
//  pipelining_ex
//
//  Created by 김도희 on 2020/02/25.
//  Copyright © 2020 김도희. All rights reserved.
//
#include<stdio.h>
//#include<Windows.h>
#include <unistd.h>
#include<pthread.h>
#include"sem.hpp"
#include"Timer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <map>
#include <fstream>
#include <sstream>
//#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;
void retina_preprocessing(Mat& img, float* img_1d);
int retina_postprocessing(float *face_rpn[9], float width_scale, float height_scale, int ws, int hs, Mat& src);

#define NUM_BUFFER 2
#define NUM_THREAD 2

int input_size = 320*320*3;
float *img_1d_0 = new float[input_size]();
float *img_1d_1 = new float[input_size]();

float *face_rpn_0[9];
float *face_rpn_1[9];

typedef struct _ThreadArgs {
    int thread_id = -1;
    int frame_num = 0;
} ThreadArgs;

sem_t sem[NUM_BUFFER];
bool b_ready_second_thread = false;


//void* buffer[NUM_BUFFER]; //FIXME
//vector<Mat> buffer;
vector<float> buffer;


vector<string> string_split(string str, char delimiter) {
    vector<string> internal;
    stringstream ss(str);
    string temp;
    
    while (getline(ss, temp, delimiter))
        internal.push_back(temp);
    
    return internal;
}


void *my_thread(void * _args) {
    ThreadArgs* args = (ThreadArgs*)_args;

    int tid = args->thread_id;
    int total_num = args->frame_num;
    int input_idx = 0;
    int buf_idx = 0;

    if (tid == 1) Start_Timer("compute");

    printf("thread %d is started.\n", tid);

    while (input_idx < total_num) {
        switch (tid) {
        case 0: /* Thread 0 */
            sem_p(&sem[buf_idx], tid, "thread 0");
            if (!b_ready_second_thread)
                b_ready_second_thread = true;

            // compute using buffer[buf_idx]
            //printf("this is thread0. it must repalce into pre/post process.\n");
            int ws = 320;
            int hs = 320;
            float width_scale = float(buffer[input_idx].rows())/320.0;
            float height_scale = float(buffer[input_idx].cols())/320.0;
                
            if (buf_idx == 0) {
                retina_preprocessing(buffer[input_idx], img_1d_0);
                if (input_idx > 1) {
                    retina_postprocessing(face_rpn_0, width_scale, height_scale, ws, hs, buffer[input_idx]);
                }
            }
            else if (buf_idx == 1) {
                retina_preprocessing(buffer[input_idx], img_1d_1);
                if (input_idx > 1) {
                    retina_postprocessing(face_rpn_1, width_scale, height_scale, ws, hs, buffer[input_idx]);
                }
            }

            sem_v(&sem[buf_idx], tid, "thread 0");
            break;

        case 1: /* Thread 1 */
            while (!b_ready_second_thread) sleep(0.01);
            sem_p(&sem[buf_idx], tid, "thread 1");

            // compute using buffer[buf_idx]
            cout << buffer[input_idx] << endl;
            sleep(2);
            //printf("this is thread1. it must repalce into inference.\n");
                PTEnv* env = new PTEnv();
                string kernel_file = "pipeline.cl";
                env->init(kernel_file);
                
                float *input = new float[env->get_layer_input_size()]();
                float *output = new float[env->get_layer_output_size(env->num_layers-1)]();
                
                int dim_in = env->param_table[0].dim_in;
                for (int c = 0; c < 3; c++) {
                    for (int y = 0; y < dim_in; y++) {
                        for(int x=0; x<dim_in; x++) {
                            int i = c*dim_in*dim_in + y*dim_in + x;
                            if(y == 0 || x == 0 || y == dim_in-1 || x == dim_in-1) input[i] = 0.f;
                            else {
                                int ii = c*320*320 + (y-1)*320 + (x-1);
                                if (buf_idx == 0)
                                    input[i] = img_1d_0[ii];
                                else
                                    input[i] = img_1d_1[ii];
                            }
                        }
                    }
                }
                
                #ifdef PT_INT8
                    int face_rpn_idx[9] = {48, 49, 50, 64, 65, 66, 80, 81, 82};
                #else
                    int face_rpn_idx[9] = {37, 38, 39, 53, 54, 55, 69, 70, 71};
                #endif
                    for(int i=0; i<9; i++) {
                        if (buf_idx == 0)
                            face_rpn_0[i] = new float[env->get_layer_output_size(face_rpn_idx[i])]();
                        else
                            face_rpn_1[i] = new float[env->get_layer_output_size(face_rpn_idx[i])]();
                    }
                int num_iters = 1;
                double t_time = 0.;
                for(int i=0; i<num_iters; i++) {
                    env->release();
                    env->init(kernel_file);

                    env->copy_input(input);

                    //gettimeofday(&start, NULL);

                    env->launch_pt();

                    //gettimeofday(&end, NULL);
                    //timersub(&end,&start,&timer);
                    //t_time += timer.tv_usec / 1000.0 + timer.tv_sec *1000.0;

                    env->copy_output(output, env->num_layers);
                }
                    //printf("GPU elapsed time (avg. %d): %lf\n", num_iters, t_time / (double)num_iters );

                for(int i=0; i<9; i++) {
                    if (buf_idx == 0)
                        env->copy_output(face_rpn_0[i], face_rpn_idx[i]+1);
                    else
                        env->copy_output(face_rpn_1[i], face_rpn_idx[i]+1);
                }
                for(int i=0; i<env->num_layers; i++)
                    env->save_layer(i);

                env->release();

                free(input);
                free(output);
                
            sem_v(&sem[buf_idx], tid, "thread 1");
            break;
        }

        //printf("thread %d: input %d is processed. (buf: %d)\n", tid, input_idx, buf_idx);

        input_idx++;
        buf_idx = (buf_idx + 1) % NUM_BUFFER;
    }

    printf("thread %d is finished\n", tid);

    if (tid == 1) End_Timerp("compute");

    return NULL;
}


int main() {
    /*
    ifstream filelist("filelist.txt");
    string s;
    vector<string> /.fi?//.le;
    while (filelist) {
        getline(filelist, s);
        file.push_back(s);
    }
    
    for (int i = 0; i < file.size(); i++) {
        vector<string> directory_parser = string_split(file[i], '/');
        vector<string> name_parser = string_split(directory_parser[directory_parser.size()-1], '.');
        float temp_buffer = float(i);
        //Mat img = imread(file[i], IMREAD_COLOR);
        buffer.push_back(temp_buffer);
        //buffer.push_back(img);
    }
    */
    for (int i = 0; i < 10; i++) {
        float temp_buffer = float(i);
        buffer.push_back(temp_buffer);
    }
    
    Init_Timer();

    pthread_t t0, t1; //thread_id

    //ThreadArgs targs0 = { 0 , (int)file.size() },
    //           targs1 = { 1 , (int)file.size() };
    ThreadArgs targs0 = { 0 , 10 },
               targs1 = { 1 , 10 };
    for (int i = 0; i < NUM_BUFFER; ++i) {
        sem_init(&sem[i], 1);
    }

    printf("thread create\n");
    pthread_create(&t0, NULL, my_thread, &targs0);     //thread 0
    pthread_create(&t1, NULL, my_thread, &targs1);     //thread 1

    pthread_join(t0, NULL); //thread 0
    pthread_join(t1, NULL); //thread 1
}
