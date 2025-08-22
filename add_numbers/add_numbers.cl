/*
Code that contains kernels to run on accelerator in parallel. A kernel 
represents the basic unit of executable code. Each kernel will be 
executed on one work item ("pixel") of your parallel task:

1 work item = 1 "pixel" in your image 

A practical application may generate thousands or even millions of 
work-items, but for the simple task of adding 64 numbers, 
eight work-items will suffice. The program file add_numbers.cl 
contains a function called add_numbers that performs this operation. 
Like all kernel functions, it returns void and its name is preceded by 
the __kernel identifier.

The kernel has 64 values to add together and eight work-items with 
which to add them. After each work-item computes its sum of eight 
values, these partial results will be added together to form a sum for 
the entire group. In the end, the kernel will return two sums — one for 
each work-group executing the kernel.

data <===== input_buffer
group_result <===== sum_buffer
*/

__kernel void add_numbers(__global float4* data, 
      __local float* local_result, __global float* group_result) {

   float sum;
   float4 input1, input2, sum_vector; // array of 4 floats which support vectorization
   uint global_addr, local_addr;

   global_addr = get_global_id(0) * 2;
   input1 = data[global_addr];
   input2 = data[global_addr+1];
   sum_vector = input1 + input2; // perform four floating-point additions simultaneously

   local_addr = get_local_id(0);
   local_result[local_addr] = sum_vector.s0 + sum_vector.s1 + 
                              sum_vector.s2 + sum_vector.s3;
   
   //printf("global id = %d, local id = %d, get_local_size(0)=%d\n", get_global_id(0), get_local_id(0), get_local_size(0));
   //printf("local_result[%d] = %f, group id = %d\n", local_addr, local_result[local_addr], get_group_id(0));

   // 同步一个工作组（work-group）中的所有工作项（work-item）。
   barrier(CLK_LOCAL_MEM_FENCE);

   if(get_local_id(0) == 0) {
      sum = 0.0f;
      for(int i=0; i<get_local_size(0); i++) {
         sum += local_result[i];
      }
      group_result[get_group_id(0)] = sum;
      //printf("group_id %d, sum = %f\n", get_group_id(0), sum);
   }
}
