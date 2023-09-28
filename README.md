# Video denoising

- **Network**
  
  - Architecture
  
    ![image-20230811222005076](./img/image-20230811222005076.png)
  
  - DTAM:可变形时间注意力模块<img src="./img/image-20230811222516152.png" alt="image-20230811222516152" style="zoom:60%;" />
  
  - DSAM：可变形空间注意力模块
  
    <img src="./img/image-20230811222749444.png" alt="image-20230811222749444" style="zoom:70%;" />
  

- Experiments

  - 去噪实验

    ![image-20230811223227456](./img/image-20230811223227456.png)

  - 分割实验

    ![image-20230811223332276](./img/image-20230811223332276.png)

## Test
```
python tools/test.py  configs/DSTAN_thyroid.py checkpoints/iter_45000_sigma300.pth  --save-path exp/denoised
```