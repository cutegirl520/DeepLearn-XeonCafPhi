/* Implicit offload of computation on Xeon-Phi 
 * Size chosen based on:
 * VSM size exceeds the limitation (17179869184) now!
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <new>

#include "CycleTimer.h"


#define MIN_VAL         (-1)
#define MAX_ITERATIONS  100

#define IN_CHANNELS     3
#define OUT_CHANNELS    10

#define IN_HEIGHT       256
#define IN_WIDTH        256

#define OUT_HEIGHT      80
#define OUT_WIDTH       80

#define KERNEL_HEIGHT   8
#define KERNEL_WIDTH    8

#define NUM             64


typedef struct Convolution {
    int conv_in_channels_;
    int conv_out_channels_;
    int conv_in_height_;
    int conv_in_width_;
    int height_out_;
    int width_out_;
    int kernel_h_;
    int kernel_w_;
    int num_;

    float *input;
    float *output;
    float *weight;
} conv;



conv *cv;
int inputSize, weightSize, outputSize, totalInputSize, totalOutputSize;

void init()
{
  int i;
  float randVal;
  
  cv = (conv *)malloc(sizeof(struct Convolution));
  cv->conv_in_channels_ = IN_CHANNELS;
  cv->conv_out_channels_ = OUT_CHANNELS;
  cv->conv_in_height_ = IN_HEIGHT;
  cv->conv_in_width_ = IN_WIDTH;
  cv->height_out_ = OUT_HEIGHT;
  cv->width_out_ = OUT_WIDTH;
  cv->kernel_h_ = KERNEL_HEIGHT;
  cv->kernel_w_ = KERNEL_WIDTH;
  cv->num_ = NUM;

  inputSize = cv->conv_in_channels_ * cv->conv_in_height_ * cv->conv_in_width_;
  weightSize = cv->conv_out_channels_ * cv->conv_in_channels_ *
                    cv->kernel_h_ * cv->kernel_w_;
  outputSize = cv->conv_out_channels_ * cv->height_out_ * cv->width_out_;
  totalInputSize = inputSize * cv->num_;
  totalOutputSize = outputSize * cv->num_; 

  cv->input = (float *)malloc(sizeof(float) * totalInputSize);
  if(cv->input == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }

  /* Read random float values */
  for(i = 0; i < totalInputSize; i++) {
    randVal = static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
    cv->input[i] = randVal;
  }
#if 0
  printf("input: \n");
  for(i = 0; i < totalInputSize; i++) {
    printf(" %f ", cv->input[i]);
  }
  printf("\n\n");
#endif

  cv->weight = (float *)malloc(sizeof(float) * weightSize);
  if(cv->weight == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }
  
  /* Read random float values */
  for