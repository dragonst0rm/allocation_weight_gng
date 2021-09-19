library('mistr')
library('copula')
library('EnvStats')

### FUNCTION TO FIT GNG DISTRIBUTION TO EACH RETURN OF ALPHA

fit_GNG<-function(ret)
  {
  ret<-na.omit(ret)
  # Find the break 1 & 2
  num1=quantile(ret,0.05,names=FALSE)
  num2=quantile(ret,0.95,names=FALSE)
  # 2 tails data and normal parts
  x<-ret[(ret>num1)&(ret<num2)]
  x1<-ret[ret<=num1]; x2<-ret[ret>=num2]
  # Find the shape parameter
  s1<-as.numeric(epareto(-x1,method='mle')$parameters[2])
  s2<-as.numeric(epareto(x2,method='mle')$parameters[2])
  # Fitting the dataset
  fit_ret<-GNG_fit(ret, start = c(break1 = num1, break2 =num2, 
                                  mean = mean(x) , sd = sd(x) ,
                                  shape1 = s1, shape2 = s2))
  cdf<-function(x){p(distribution(fit_ret),x)}
  icdf<-function(x){q(distribution(fit_ret),x)}
  return (list(cdf,icdf,fit_ret))}


### SIMULATION FROM GAUSSIAN AND CLAYTON COPULA

COPULA<- function(M,N,list_cdf, list_icdf)
{
  n<- ncol(M) ; m<-nrow(M); X<-matrix(0,m,n)
  # Turn return to Uniform data
  for (i in 1:n){X[,i]<-list_cdf[[i]](M[,i])}
  
  # Gaussian Copula
  fit_Gaussian  <-fitCopula(normalCopula(dim=n), X, method = 'ml')
  Gaussian_model<-normalCopula(coef(fit_Gaussian) ,dim=n)
  
  # Simulation
  U_G<- rCopula(N,Gaussian_model)
  
  #Turn sim_U to sim_return
  X_G<-matrix(0,N,n) 
  for (i in 1:n)
  {X_G[,i]<-list_icdf[[i]](U_G[,i]) }
  
  return(list(X_G)) }


### REMOVE OUTLINER FUNCTION

remove_outliner<-function(M,data)
{
  n<-ncol(data)
  for(i in 1:n)
  {
    min_ret<-min(M[,i]); max_ret<-max(M[,i])
    data<-data[(data[,i]>=min_ret)&(data[,i]<=max_ret),]
  }
  return(data)
}



M<-read.csv('/home/hoainam/PycharmProjects/multi_strategy/TEST_GNG/Train/train9.csv')

# We have a total of 10 alphas
M<-as.matrix(M[,2:11]); n<- ncol(M)

# Find cdf and icdf function
list_cdf<-list() ; list_icdf<-list()
for (i in 1:n)
{func_list<-fit_GNG(M[,i])
  list_cdf[[i]]<-func_list[[1]]; list_icdf[[i]]<-func_list[[2]]}

# Simulation from Copula
N<-2000000
X<-COPULA(M,N,list_cdf, list_icdf)

# We use Gaussian Copula 
X[[1]]<-remove_outliner(M,X[[1]])
X<-as.data.frame(X[[1]]); dimnames(X)[[2]]<-dimnames(M)[[2]]


write.csv(X,'/home/hoainam/PycharmProjects/multi_strategy/TEST_GNG/simulate/X_G_train9.csv')


