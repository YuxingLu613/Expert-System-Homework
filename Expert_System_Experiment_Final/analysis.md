### 读取并分析数据

操作概述：通过pandas库读取csv文件，使用matplotlib画出各类元素的分布情况，再根据分布图进行分析。

 

商品价格分布：

![img](file:///C:\Users\Colton\AppData\Local\Temp\ksohtml2760\wps1.jpg) 

分析：存在一些数值较大的干扰值，在清洗数据时应将item price限制在10,000以下。

 

商品销量分布：

![img](file:///C:\Users\Colton\AppData\Local\Temp\ksohtml2760\wps2.jpg) 

分析：存在一些数值较大的干扰值，在清洗数据时应将item cnt day限制在1,000以下。

 

月份分布：

![img](file:///C:\Users\Colton\AppData\Local\Temp\ksohtml2760\wps3.jpg) 

分析：每个月份均有售卖的商品，无需进行特别的处理。