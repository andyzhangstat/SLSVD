# Sim3: 3D, Normal, Overlapping (vary SNR to generate a figure)
# vs CP
library(GLMMadaptive)
rec_SNR <- matrix(0, nrow = 10, ncol = 4) # theta recovery (smaller=better)
colnames(rec_SNR) <- c("SNR", "CORALS", "CP", "temp")
rec_nonzero <- array(0, dim = c(2, 100, 100)) # sensitivity/specificity (larger=better)
rec_theta <- matrix(0, nrow = 100, ncol = 2) # theta recovery (smaller=better)
colnames(rec_theta) <- c("CORALS", "CP")
rec_time <- matrix(0, ncol = 2, nrow = 100)
colnames(rec_time) <- c("CORALS", "CP")
set.seed(20190312)
p <- c(50, 50, 50)
V1true <- cbind(sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4), 
                rep(0, p[1] - 20),
                c(rep(0, 30), sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4), 
                  rep(0, p[1] - 50)))
V2true <- cbind(sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4), 
                rep(0, p[2] - 20),
                c(rep(0, 30), sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4), 
                  rep(0, p[2] - 50)))
V3true <- cbind(sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4), 
                rep(0, p[3] - 20),
                c(rep(0, 30), sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4), 
                  rep(0, p[3] - 50)))
V1true <- norm_mat(V1true) * diag(c(80, 50))
V2true <- norm_mat(V2true)
V3true <- norm_mat(V3true)
Thetatrue <- TensProd_GL(list(V1true, V2true, V3true), 1:2)
truecc <- (Thetatrue != 0)

for (ind in 1:length(SNR_cand)) {
  SNR <- SNR_cand[ind]
  sigma <- sqrt(var(Thetatrue) / SNR)
  
  print(paste("preset SNR:", SNR))
  


  for (i in 1:nsim) {
  # generate data
  set.seed(Sys.time())
  X <- matrix(rnorm(prod(p), Thetatrue, sigma), ncol = p[1])
  
  # CORALS
  tic()
  Va <- GCC(X, type = "normal", numBC = 2, fig = FALSE)
  Theta1 <- TensProd_GL(Va, 1:2)
  estcc <- (Theta1 != 0)
  temp1 <- estcc & truecc
  temp2 <- (estcc == 0) & (truecc == 0)
  sen1 <- sum(temp1) / sum(truecc)
  spc1 <- sum(temp2) / sum(1 - truecc)
  T1 <- toc()
  
  # CP
  tic()
  Vb <- parafac_GL(X, 2)
  Theta2 <- TensProd_GL(Vb, 1:2)
  T2 <- toc()
  
  # record results
  rec_nonzero[, i, ] <- cbind(sen1, spc1)
  rec_theta[i, ] <- c(norm(Theta1 - Thetatrue, "F"), norm(Theta2 - Thetatrue, "F"))
  rec_time[i, ] <- c(T1, T2)
  X_forR[, , , i] <- array(X, dim = p)
}
rec_SNR <- cbind(rec_SNR, rec_theta) # every 2 columns form a group







## Sim4: 3D, count, overlapping
## vs Poisson CP
## poisson tensor, rank-2 biclusters

p <- c(50, 50, 50)
set.seed(12102018)
V1true <- cbind(sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4), 
                rep(0, p[1] - 20),
                c(rep(0, 10), sign(runif(30) - 0.5) * (runif(30) * 0.1 + 0.4), 
                  rep(0, p[1] - 40)))
V2true <- cbind(sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4),
                rep(0, p[2] - 20),
                c(rep(0, 10), sign(runif(30) - 0.5) * (runif(30) * 0.1 + 0.4), 
                  rep(0, p[2] - 40)))
V3true <- cbind(sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4),
                rep(0, p[3] - 20),
                c(rep(0, 20), sign(runif(20) - 0.5) * (runif(20) * 0.1 + 0.4), 
                  rep(0, p[3] - 40)))
V1true <- norm_mat(V1true) * diag(c(50, 40))
V2true <- norm_mat(V2true)
V3true <- norm_mat(V3true)
Thetatrue <- TensProd_GL(list(V1true, V2true, V3true), 1:2) + 2 # add mean shift because Poisson data are skewed (low signal for negative Theta)
truecc <- (Thetatrue != 2)
Lambda <- exp(Thetatrue)

## run 100 simulations, with the same Theta
nsim <- 100
rec_nonzero <- matrix(0, nrow = nsim, ncol = 2) # sensitivity/specificity (larger=better)
rec_theta <- matrix(0, nrow = nsim, ncol = 2) # theta recovery (smaller=better)
rec_time <- matrix(0, nrow = nsim, ncol = 2)


for (i in 1:nsim) {
  set.seed(Sys.time()) # generate data
  X <- rpois(prod(p), Lambda) %>% matrix(nrow = p[[1]]) # convert to tensor
  hist(X) # plot histogram of data
           
  # CORALS
  Va <- GCC(X, type = 'poisson', options = list(numBC = 4, fig = 0))
  Theta1cc <- TensProd_GL(Va, 2:3) # only 2:3 has sparsity; layer1 capture mean shift
  estcc <- (Theta1cc != 0)
  temp1 <- estcc & truecc
  temp2 <- (estcc == 0) & (truecc == 0)
  sen1 <- sum(temp1) / sum(truecc)
  spc1 <- sum(temp2) / sum(1 - truecc)
  Theta1 <- TensProd_GL(Va)
  T1 <- system.time({
    GCC(X, type = 'poisson', options = list(numBC = 4, fig = 0))
  })[3]
  
  # logX CP
  logX <- log(pmax(X, 0.5))
  Vb <- parafac_GL(logX, 3)[[1]]
  Theta2 <- TensProd_GL(Vb)
  T2 <- system.time({
    parafac_GL(logX, 3)
  })[3]
  
  # record results
  rec_nonzero[i, ] <- c(sen1, spc1)
  rec_theta[i, ] <- c(norm(Theta1 - Thetatrue, "F"), norm(Theta2 - Thetatrue, "F"))
  rec_time[i, ] <- c(T1, T2)
}















