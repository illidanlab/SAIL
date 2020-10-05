## Self-Adaptive Imitation Learning (SAIL)

##### Code Implementations for: 
- The proposed Algorithm: 
    - [Learning Sparse Rewarded Tasks from Sub-Optimal Demonstrations](https://arxiv.org/pdf/2004.00530v1.pdf)
- As well as other approaches:
    -  [Discriminator Actor Critic (DAC)](https://arxiv.org/abs/1809.02925)
    <!--- - [Policy Optimization from Demonstrations (POfD)](http://proceedings.mlr.press/v80/kang18a/kang18a.pdf) ---> 
    - [Generative Adversarial Imitation Learning (GAIL)](https://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf)
 ---- 
##### Please be noted that:
- All algorithms are implemented based on the **stable-baseline** code framework: https://stable-baselines.readthedocs.io/en/master/
 --- 
 
##### Installation:
Please also check the stable-baselines webpage for installing preliminary packages: https://github.com/hill-a/stable-baselines
   <pre><code>cd stable_baselines
   pip install -e . </code></pre> 
 
##### How to run:
-   Run SAIL on environment HalfCheetah-v2, running seed = 3, using 1 teacher trajecotries:
    <pre><code>cd stable_baselines/run
    python train_sail.py --env HalfCheetah-v2 --seed 3 --algo sail --log-dir your/log/dir/ --task gail-lfd-adaptive-dynamic --n-timesteps -1 --n-episodes 1 </code> </pre>
    - Results will be written to <code>your/log/dir/gail-lfd-adaptive/sail/HalfCheetah-v2/rank3/ </code>.
    - **Note** that: the <code>task</code> option must contain substrings of *gail*, *lfd*, *adaptive*, and *dynamic*.
        - *lfd*: learning from teacher demonstrations
        - *gail*: use adversarial training.
        - *adaptive*: replace teacher buffer with better trajectories.
        - *dynamic*: turn off mixture sampling by turning <img src="https://render.githubusercontent.com/render/math?math=\alpha = 0"> after the student has surpassed the teacher (see details in the paper).
        
- Run DAC on environment Hopper-v2, running seed=5, using 4 teacher trajctory:
    <pre><code>  cd stable_baselines/run
    python train_sail.py --env Hopper-v2 --seed 5 --algo dac --log-dir your/log/dir/ --task dac-gail --n-timesteps -1  --n-episodes 4 </code> </pre>
    
 
- Run GAIL on Swimmer-v2, running seed = 2, using 1 teacher trajectory:   
    <pre><code>  cd stable_baselines/run
    python train_sail.py --env Swimmer-v2 --seed 2 --algo trpo --log-dir your/log/dir/ --task trpo-gail --n-timesteps -1 --n-episodes 1 </code> </pre>
     


##### Reminders:
- Teacher demonstrations are saved at <code>SAIL-code/teacher_dataset/</code>. For each environment, we collected 1, 4, 10 trajectories from a sub-optimal teacher.
- Hyper-parameters can be found at <code>SAIL-code/stable-baselines/hyperparams/</code>.
    - You may need to tune those parameters to fit your machine.
  
