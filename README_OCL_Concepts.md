# OCL 编程相关 Concepts

#### __local 与 __global 和 __private 的区别 <br>

|内存类型 |	关键字 | 作用范围 |	访问速度 |	典型用途|
| ------ | -----  | ------- | -------- | -------- |
|本地内存|	__local   |	同一个工作组内的所有工作项 |	较快 |	工作组内共享数据，减少全局内存访问
|全局内存|	__global  |	所有工作组和主机	      |    较慢 |  输入/输出数据，设备与主机间通信, 一般指L3.
|私有内存|	__private |	单个工作项	              |   最快	|  存储工作项的私有变量，如循环计数器


``工作项：``work-item, 对应一个具体的元素的计算。 <br>
``工作组：``group，包含很多work-item, 每个group内，__local 变量共享。 <br>

使用```clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL)```设置global size和local size。 <br>
例如：
```
global_size = 8; 
local_size = 4; 
num_groups = global_size/local_size;  // group size
```

#### 同步

```barrier(CLK_LOCAL_MEM_FENCE)``` 在 OpenCL kernel 中的作用是同步一个工作组（work-group）中的所有工作项（work-item）。