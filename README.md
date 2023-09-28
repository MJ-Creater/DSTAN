- **Network**
  
  - Architecture
  
    ![image-20230811222005076](./img/image-20230811222005076.png)
  
  - DTAM:可变形时间注意力模块<img src="./img/image-20230811222516152.png" alt="image-20230811222516152" style="zoom:60%;" />
  
  - DSAM：可变形空间注意力模块
  
    <img src="./img/image-20230811222749444.png" alt="image-20230811222749444" style="zoom:70%;" />
  

- Experiments

  - 去噪实验

    ![image-20230811223227456](./img/0ccffb1318cee1e36e244bdbae0ff2b.png)

  - 分割实验

    ![image-20230811223332276](./img/0b0c448be05976e476ee671a3887a12.png)

## Test
```
python tools/test.py configs/DSTAN_thyroid.py checkpoints/iter_45000_sigma300.pth --save-path exp/denoised
```
