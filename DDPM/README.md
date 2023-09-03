[Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/pdf/2006.11239.pdf)       

<p align='center'><img src='https://github.com/WestChaeVI/Diffusion_Models/assets/104747868/4a5ac0ed-4966-4efb-8875-3301f6d2a5b9'></p>       

------------------------------           

## Introduction    

+ 본 논문은 diffusion probabilistic models의 progress를 소개한다.    

+ **Diffusion model은 data에 noise를 조금씩 더해가거나 noise로부터 조금씩 복원해가는 과정**을 통해 **data를 generate하는 model**이다.    

  - 주어진 iamge에 time $t$에 따른 상수의 파라미터를 갖는 작은 Gaussian noise를 $t$에 대해 더해나가는데, image가 destroy하게 되면 결국 noise의 행태로 남을 것이다. (normal distribution을 따른다.)    

  - 이런 상황에서 **normal distribution에 대한 noise가 주어졌을 때, 어떻게 복원할 것인가에 대한 문제**이다.    

  - 그래서 **주어진 noise를 통해 이미지를 완전히 복구**하게 된다면 **image generation**이 되는 것이다.    

+ diffusion model은 **유한한 시간 뒤에 이미지를 생성하는 variational inference**를 통해 **훈련된 Marcov chain을 parameterized한 형태**이다.    
  > Marcov chain은 이전의 smapling이 현재 sampling에 영향을 미치는 $p(x|x)$ 형식을 의미한다.      

+ 이를 한 눈에 표현하면 아래 그림과 같다. 아래 그림에서 $x_0$는 실제 데이터, $x_T$는 최종 noise, 그리고 그 사이의 $x_t$는 데이터에 noise가 더해진 상태의 latent variable을 의미한다.    

<img width="1000" alt="스크린샷_2023-02-20_오후_4 28 54" src="https://github.com/WestChaeVI/Diffusion_Models/assets/104747868/7a43587d-f433-4c2c-accd-1743cabb7cc0">    

+ 위 그림의 우측에서 좌측 방향으로 noise를 점점 더해가는 *forward process* **$q(x_{t}|x_{t-1})$** 를 진행한다.      
그리고 이 *forward process*를 반대로 추정하는 *reverse process* **$p_{\theta}(x_{t-1}|x_{t})$** 를 학습함으로써 $noise(x_T)$ 로부터 $data(x_0)$ 를 복원하는 과정을 학습한다.     

+ 이 **reverse process**를 활용해서 random noise로부터 우리가 원하는 **image, text, graph**등을 generate할 수 있는 모델을 만들어내는 것이다.    

------------------------------           

## Background    

### 1. Forward Diffusion Process, $q(\cdot)$      

<p align='center'><img src='https://github.com/WestChaeVI/Diffusion_Models/assets/104747868/3b315da2-5c32-432c-9448-1bef3c7ed6c2'></p>      

+ 주어진 iamge를 $x_0$라고 하고, 서서히 noise를 추가해가는 과정을 $q(\cdot)$ 라고 해보자.    

+ 그럼 $x_0$에 noise를 적용해서 $x_1$를 만드는 것을 **$q(x_1|x_0)$** 이라고 표현할 수 있다.    

  - 위 표현을 time $t$에 대해 general하게 표현한다면 **$q(x_t|x_{t-1}$** 으로 표현할 수 있다.    

  - 이를 forward process(or diffusion process)라고 부른다.     

+ 이 때, time $t$가 끝까지 간다면 $x_T$, 즉 완전히 destroy 된 형태가 나온다. 이는 normal distribution $\mathcal{N}(x_T \ ; \ 0, \ I)$ 를 따른다.    

+ 정리하자면, **Forward process, $q(\cdot)$** 는 Marcov chain으로 $data(x_0)$로부터 noise를 더해가면서 최종 $noise(x_T)$ 형태로 가는 과정이다.     
  - **우리가 이 과정의 분포를 알아내야 하는 이유는, reverse process의 학습을 forward process의 정보를 활용하기 때문**이다.      


<details>
<summary>Forward process의 자세한 수식</summary>

$$q(\mathrm{x}\_{1:T} | \mathrm{x}\_0) \ := \ \prod_{t=1}^{T} q(\mathrm{x}\_t | \mathrm{x}\_{t-1}), \ \ \ q(\mathrm{x}\_{t} | \mathrm{x}\_{t-1}) \ := \ \mathcal{N}(\mathrm{x}\_t \ ; \ \sqrt{1 \ - \ \beta_t} \mathrm{x}\_{t-1}, \ \beta_t I)$$      

+ data에 noise를 추가할 때, **variance schedule $[\beta_t \in (0, 1) \]_{t=1}^T$ 를 이용하여 scaling한 후 더해준다**.     

  - 매 step마다 gaussian distribution에서 reparameterize를 통해 smapling하게 되는 형태로 noise가 추가되는데, 이때 **단순히 noise만을 더해주는 것이 아니라 $\sqrt{1 \ - \ \beta_t}$ 로 scaling하는 이유는 variance가 발산하는 것을 막기 위함이다.**    

  - variance를 unit하게 가둠으로써 forward-reverse 과정에서 variance가 일정 수준으로 유지될 수 있게 된다.    

    $$x_t \ = \ \sqrt{1 \ - \ \beta_t} x_{t-1} \ + \ \beta_t \cdot \epsilon$$

    $$(\sqrt{1 \ - \ \beta_t} x_{t-1})^2 \ + \ \beta_t \ = \ 1$$       

  - 이 값은 **learnable parameter로 둘 수도 있지만, 저자는 실험을 통해 상수로 둬도 큰 차이가 없다**는 결과를 도출해냈고, 결국 **constant로 두었다고 한다**.     
    > 데이터가 이미지와 비슷할 때에는 이 값을 매우 작게 설정하다가 gaussian distribution에 점점 가까워질수록 이 값을 크게 설정 ($10^{-4}$에서 0.02로 linear하게 증가)    

+ time step $t$번의 sampling을 통해 매 step을 차근차근 밟아가면서 $x_0$에서 $x_t$를 만들 수도 있지만, 한번에 이를 할 수도 있다.    

  - 재귀적으로 식을 정리하다보면, 어떤 data $x_0$가 주어졌을 때 $x_t$의 분포는 다음과 같다.    
    $$\alpha_t \ := \ 1 - \beta_t \ \ \ \ \bar{\alpha_t} \ := \ \prod_{s=1}^t \alpha_s$$

    $$q(x_t | x_0) \ = \ \mathcal{N}(x_t \ ; \ \sqrt{\bar{\alpha_t}} x_0, \ (1 - \bar{\alpha_t}) I)$$     

  - 한 step씩 학습을 하면 메모리와 resource가 매우 많이 든다. 그러나 위와 같이 한번에 $q(x_t | x_0)$ 를 만들고 나면, 여기서 loss를 구한 다음, $t$에 대한 expectation을 구하는 식으로 학습이 가능하다.     

+ 아래 그림은 위의 식 전개에 대한 증명이다.     

<p align='center'><img src='https://github.com/WestChaeVI/Diffusion_Models/assets/104747868/8d71edbf-ed55-48fa-8792-b229ee6a4638'></p>      

</details>       


### 2. Reverse Diffusion Process, $p(\cdot)$     

+ **Reverse process**는 noise를 점점 더해나가는 $q(\cdot)$와는 반대로 **noise를 점진적으로 걷어내는 denoising process이다**.     

+ 최종적으로 random noise로부터 data를 generate하는 generative model로 사용되기 때문에 diffusion model을 사용하기 위해서는 모델링하는 것이 필수적이지만, 이를 실제로 알아내는 것은 쉽지 않다.      

  - 우리가 알고 싶은 것 : $q(x_{t-1} | x_t)$     

    > 그러나, **각 $t$시점에서 이미지의 확률 분포 $q(xt)$를 알 수 없기 때문에 베이즈 정리에 의해 계산되지 않는다.**     
    >       
    > **때문에 확률분포 $q(\cdot)$가 주어졌을 때, 이 확률 분포를 가장 잘 모델링하는 확률 분포 $p_\theta$를 찾는 문제로 변환 한다.**     

<details>
<summary>Reverse process의 자세한 수식</summary>

+ 즉, 우리는 $p_\theta$ 를 활용해서 이를 approxiamte한다. 이때, 이 approximation은 Gaussian transition을 활용한 Markov chain의 형태를 가진다.     

  - 이를 식으로 표현하면 다음과 같다.     

    $$p_\theta (\mathrm{x}\_{0:T} \ := \ p(\mathrm{x}\_T) \prod_{t=1}^T p_\theta(\mathrm{x}\_{t-1} | \mathrm{x}\_t), \ \ \ \ p_\theta (\mathrm{x}\_{t-1} | \mathrm{x}\_t) \ := \ \mathcal{N} ( \mathrm{x}\_{t-1} \ ; \ \mu_\theta (\mathrm{x}\, t), \sum\nolimits_{\theta} ( \mathrm{x}\_t , t))$$    

+ 위 식에서, 각 step의 정규 분포의 평균 $\mu_\theta$와 표준편차 $\sum\nolimits_\theta$ 는 학습되어야 하는 parameter들이다. 위 식의 시작 지점인 noise의 분포는 다음과 같이 가장 간단한 형태의 표준정규분포로 정의한다.    

$$p(x_t) \ = \ \mathcal{N}(x_T \ ; \ 0, \ I)$$     

</details>       


### 3. Objective Function      

+ 이제 $p_\theta$의 parameter estimation을 위해 diffusion model을 어떻게 학습시키는지에 대해 알아보자.     

+  **$x_t$가 들어왔을 때 $x_{t-1}$을 예측할 수 있게 된다면, 우리는 $x_0$ 또한 예축할 수 있다.**

+ 앞서 언급했듯이 **실제 data의 분포의 $p_\theta(x_0)$를 찾아내는 것을 목적**으로 하기 때문에 **결국 이의 likelihood를 최대화(Negative likelihood를 minimize)하는 것이 우리가 원하는 목적**이다.     

  - 이를 수식으로 나타내면 다음과 같다. (Diffusion model의 Loss function)           

    $$\mathbb{E}\left\[-\log p_\theta (\mathrm{x}\_0)\right\] \le \mathbb{E}\_q\left\[ -\log \frac{p_\theta(\mathrm{x}\_{0:T})}{q(\mathrm{x}\_{1:T} | \mathrm{x}\_0)} \right\] \ = \ \mathbb{E}\_q\left\[ -\log p(\mathrm{x}\_T) - \sum_{t=1} \log \frac{p_\theta (\mathrm{x}\_{t-1} | \mathrm{x}\_t)}{q(\mathrm{x}\_t | \mathrm{x}\_{t-1})} \right\] =: \mathcal{L}$$      

<details>
<summary>증명 보기</summary>      
 
-       
  $-\log p_\theta (x_0) \ \le \ -\log p_\theta (x_0) \ + \ D_{KL} (q(x_{1:T} \ | \ x_0) \  || \ p_\theta (x_{1:T} \ | \ x_0))$           
  
  $= -\log p_\theta(x_0) + \mathbb{E}\_{x_{1:T} \sim q(x_{1:T} | x_0)}\left\[\log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T}) / p_\theta(x_0)} \right\]$     
  
  $= -\log p_\theta(x_0) + \mathbb{E}\_{x_{1:T} \sim q(x_{1:T} | x_0)}\left\[\log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} \ + \ \log p_\theta (x_0)\right\]$     
  
  $= \mathbb{E}\_q \left\[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} \right\]$       
  
  $= \mathbb{E}\_q \left\[ -\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right\]$      

</details>       

+ 또한 training loss의 세 번째 등호는 우리가 **reverse process와 forward process를 Markov chain으로 정의**했기 때문에 **Markov property에 의해 성립**한다.    
  > **Markov property**      
  >       
  > 어떤 시간에 특정 state에 도달하든 그 이전에 어떤 state를 거쳐왔든 다음 state로 갈 확률은 항상 같다는 성질      

+ 마지막으로, 위 식을 좀 더 쉽게 계산하기 위해 다음과 같은 Gaussian distribution 간의 KL divergence 형태로 식을 변형한다.     


$$\mathbb{E}\_q \left\[\underbrace{D_{KL}(q(\mathrm{x}\_T | \mathrm{x}\_0) || p(\mathrm{x}\_T))}\_{\mathcal{L}\_T} + \sum_{t > 1} \underbrace{D_{KL}(q(\mathrm{x}\_{t-1} | \mathrm{x}\_t, \mathrm{x}\_0) || p_\theta(\mathrm{x}\_{t-1} | \mathrm{x}\_{t}))}\_{\mathcal{L}\_{t-1}} - \underbrace{\log p_\theta(\mathrm{x}\_0 | \mathrm{x}\_1)}_{\mathcal{L}\_0}\right]$$     

+ 위 식에서 각각의 term이 가지는 의미를 하나씩 살펴보면 다음과 같다.     

  - $\mathcal{L}\_T$ : $p(\cdot)$가 generate하는 $noise(x_T)$와 $x_0$라는 데이터가 주어졌을 때 $q(\cdot)$가 만들어내는 $noise(x_T$ 간의 분포 차이    

    > $$q(x_t | x_0) \ = \ \mathcal{N}(x_t ; \ \sqrt{\bar{\alpha_t}} x_0, \ (1 - \bar{\alpha_t})I)$$     
    >       
    > $$p(x_T) \ = \ \mathcal{N}(x_T; \ 0, \ I)$$      


  - $\mathcal{L}\_{t-1}$ : reverse / forward process의 분포 차이. 이들의 분포 차이를 최대한 줄이는 방향으로 학습한다.    

    > $q(x_{t-1} | x_t)$는 알 수 없지만, $q(x_{t-1} | x_t, \ x_0)$은 알 수 있다.     
    > - by Bayes' rule $\rightarrow$ posterior, prior 사용    
    >      
    >  - $P(x_{t-1} | x_t) \ = \ \frac{P(x_t | x_{t-1} P(x_{t-1}}{P(x_t)}$      
    >       
    > $p_\theta(x_{t-1} | x_t) \ := \ \mathcal{N}(x_{t-1} ; \ \mu_{theta}(x_t, \ t), \sum\nolimits_{\theta} (x_t, \ t))$    

  - $\mathcal{L}\_{0}$ : latent $x_1$으로부터 data $x_0$를 추정하는 likelihood, 이를 maximize하는 방향으로 학습한다.    
 
<details>
<summary>증명 보기</summary>     

<p align='center'><img src='https://github.com/WestChaeVI/Diffusion_Models/assets/104747868/9347de90-9f20-4d1f-bfb3-604a1d381f66'></p>      

</details>     
 
+ forward process에 대한 정보를 가지고 있고 forward process의 posterior는 reverse process와 연관이 깊은 형태이기 때문에 tractable하다.   

+ 위 수식들에 대해서 좀 더 깊이 있게 들어가보자.    

------------------------------           

## Diffusion models and denoising autoencoders     

### Objective Function, $\mathcal{L}$     

$$\mathbb{E}\_q \left\[\underbrace{D_{KL}(q(\mathrm{x}\_T | \mathrm{x}\_0) || p(\mathrm{x}\_T))}\_{\mathcal{L}\_T} + \sum_{t > 1} \underbrace{D_{KL}(q(\mathrm{x}\_{t-1} | \mathrm{x}\_t, \mathrm{x}\_0) || p_\theta(\mathrm{x}\_{t-1} | \mathrm{x}\_{t}))}\_{\mathcal{L}\_{t-1}} - \underbrace{\log p_\theta(\mathrm{x}\_0 | \mathrm{x}\_1)}_{\mathcal{L}\_0}\right]$$     

+ 여기서 **가장 중요하게 봐야할 term은 바로 $\mathcal{L}_{t-1}$이다**.        

  - $x_0$부터 시작하여 conditional하게 식을 전개하다보면, tractable한 forward process posterior $q(x_{t_1} | x_t, \ x_0)$의 정규분포를 알 수 있는데, 이를 바탕으로 KL divergence를 계산하면 우리가 결과적으로 학습하고자 하는 $p_\theta(x_{t-1} | x_t)$를 학습시킬 수 있다.    

### 1. Forward Diffusion Process, $\mathcal{L}_T$       

$$\mathcal{L}\_T \ = \ D_{KL} (q(x_T | x_0) \ || \ p(x_T))$$      

+ **Objective function의 첫 번째 term**     

+ 논문에서는 forward process variances인 $\beta$를 learnable한 parameter로 두는게 아니라 상수로서 Fix하기 때문에 $\mathcal{L_T}$는 고려하지 않아도 된다.    

  - Why?      
    > DDPM에서의 forward process는 $x_T$가 항상 gaussian distribution을 따르도록 하기 때문에 사실상 ***tractable한 distribution*** $q(x_T | x_0)$ ***는 prior*** $p(x_T)$ ***와 거의 유사하다*** .     
    >      
    > 또한, DDPM에서는 forward process variance를 상수로 고정 시킨 후 approximate posterior를 정의하기 때문에 이 posterior에는 learnable parameter가 없다.     

+ 따라서 **이 loss term($\mathcal{L}_T$)은 항상 0에 가까운 상수이며, 학습과정에서 무시된다.**      


### 2. Reverse Diffusion Process, $\mathcal{L}_{1:t-1}$       

$$\mathcal{L}\_{t-1} \ = \ D_{KL} (q(x_{t-1} | x_t, \ x_0) \ || \ p_{\theta} (x_{t-1} | x_t ))$$      

+ **Objective function의 두 번째 term**    

+ 이를 계산하기 위해서 첫 번째로 $q(x_{t-1} | x_t, \ x_0)$의 분포를 알아내고, 두 번째로 $p_\theta(x_{t-1} | x_t)$를 알아내기 위해 $\sum\nolimits_{\theta}$ 와 $\mu_\theta$ 를 알아내야 한다.     

  - $\text{1}$. $q(x_{t-1} | x_t, \ x_0)$     
   
    > $q(x_{t-1} | x_t, \ x_0) \ = \ \mathcal{N}(x_{t-1} ; \ \tilde{\mu}(x_t, \ x_0), \ \tilde{\beta_t}I)$     
    >      
    > $where, \ \tilde{\mu}(x_t, x_0) \ := \ \frac{\sqrt{\bar{\alpha}\_{t-1}\beta_t}}{1-\bar{\alpha}\_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t, \ \tilde{\beta}\_t \ := \ \frac{(1 - \bar{\alpha}\_{t-1})}{1 - \bar{\alpha}_t}\beta_t$       

  - $\text{2}$. $p_\theta(x_{t-1} | x_t)$      
  
    > $p_\theta(x_{t-1} | x_t) \ = \ \mathcal{N}(x_{t-1} ; \ \mu_\theta(x_t, \ t), \ \sum\nolimits{\theta} (x_t, \ t))$    

  - $\text{3}$. $\sum\nolimits_{\theta}$    

    > 다음으로, **$p(\cdot)$의 표준편차**는 $\sigma_t^2 I$ 라는 **상수 행렬로 정의한다. 그러므로 이에 대해서는 학습이 필요하지 않다**. ( $\sigma_t^2$ 를 $\tilde{\beta}_t$로 표현하는게 맞지만 $\beta$로 사용해도 무방)      
    >      
    > $\sum\nolimits_{\theta} (x_t, \ t) \ = \ \sigma_t^2 I$     


  - $\text{4}$. $\mu_\theta(x_t, \ t)$     

    > $p(\cdot)$의 평균 $\mu_theta (x_t, \ t)$ 는 다음과 같이 정의한다.    
    >      
    > $\mu_\theta(x_t, \ t) \ = \ \tilde{\mu}\_t \left\( x_t, \ \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}\_t} \epsilon\_theta (x_t)) \right\) \ = \ \frac{1}{\sqrt{\alpha_t}} \left\( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}\_t}} \epsilon\_theta (x_t, \ t) \right\)$     
    >      
    > $\mathbb{E}\_{x_0, \ \epsilon} \left\[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \lVert \epsilon \ - \ \epsilon\_theta ( \sqrt{\bar{\alpha}_t} x_0 \ + \ \sqrt{1 - \bar{\alpha}_t} \epsilon, \ t ) \rVert^2 \right\]$     
    >      
    > <p align='center'><img src='https://github.com/WestChaeVI/Diffusion_Models/assets/104747868/acb7500f-7b3a-405e-adc1-d3630e6bde7b'></p>  


------------------------------           




------------------------------           







------------------------------           






------------------------------           
