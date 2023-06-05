

# % This is the function for generalized co-clustering analysis.
# % It employs a deflation algorithm and estimate one layer at a time
# % (treating previous layers as offset). Loadings are sparse.
# %
# %
# % input: 
# %
# %   data    p1*p2*...pK data array from some exponential family distribution
# %
# %   distr   string, specify the distribution, 
# %            choose from 'bernoulli','poisson','normal'
# %            (currently not available for other distributions)
# %
# %
# %   paramstruct
# %
# %           Tol_outer   converging threshold for max principal angle
# %                       between consecutive loading estimates, default = 0.1 degree
# %
# %           Tol_inner   converging threshold for inner iterations of IRLS
# %                       for estimate of each loading, default = 0.1 degree
# %
# %           Niter_outer max number of outer alternations (between different loadings),
# %                       default=200
# %
# %           Niter_inner max number of inner IRLS,  default=50
# %
# %           numBC       number of coclusters to identify
# %                       default=1 (i.e., only find the first cocluster)
# %
# %           fig         scalar, 0(default) no figure output
# %                       1 show tuning selection figures over iterations
# %
# % Output: 
# %
# %   V       length-K list, V{k} is a pk by numBC sparse loadings in the k
# %           dimention. Not strictly orthogonal; unit norm other than 1st
# %           array
# %
# % Originated on 8/26/2018 by Gen Li




GCC <- function(data,distr, paramstruct) { 

p <- dim(data) 
K <- length(p) # number of dimensions Index <- 1:K

initial values
Tol_outer <- 0.1 # overall threshold 
Tol_inner <- 0.1 # threshold for each component 
Niter_outer <- 200 # max number of alternations between u and v 
Niter_inner <- 50 # max number of IRLS iterations 
numBC <- 1 # number of biclusters 
fig <- 0 # whether to show figures 
stophere <- 0 # early termination sign 
if (missing(paramstruct)) { paramstruct <- list() } 
if (!is.null(paramstructTol_outer)) { Tol_outer <- paramstructTol_outer } 
if (!is.null(paramstructTol_inner)) { Tol_inner <- paramstructTol_inner } 
if (!is.null(paramstructNiter_outer)) { Niter_outer <- paramstructNiter_outer } 
if (!is.null(paramstructNiter_inner)) { Niter_inner <- paramstructNiter_inner } 
if (!is.null(paramstructnumBC)) { numBC <- paramstructnumBC } 
if (!is.null(paramstructfig)) { fig <- paramstructfig }

define critical functions for exponential family (entrywise calc)
if (distr=='bernoulli') { 
fcn_b <- function(theta) log(1+exp(theta)) 
fcn_g <- function(mu) log(mu/(1-mu)) 
fcn_ginv <- function(eta) exp(eta)/(1+exp(eta)) 
fcn_db <- function(theta) exp(theta)/(1+exp(theta)) 
fcn_ddb <- function(theta) exp(theta)/((1+exp(theta))^2) 
fcn_dg <- function(mu) 1/(mu*(1-mu)) 
pseudodata <- log((data0.8+0.1)/(0.9-0.8data)) # for initialization (0,1->0.1,0.9) } 
else if (distr=='poisson') { 
fcn_b <- function(theta) exp(theta) 
fcn_g <- function(mu) log(mu) 
fcn_ginv <- function(eta) exp(eta) 
fcn_db <- function(theta) exp(theta) 
fcn_ddb <- function(theta) exp(theta) 
fcn_dg <- function(mu) 1/mu 
pseudodata <- log(data+0.1) # for initialization (shift->0.1 to avoid 0) } 
else if (distr=='normal') { 
fcn_b <- function(theta) (theta^2)/2 
fcn_g <- function(mu) mu 
fcn_ginv <- function(eta) eta 
fcn_db <- function(theta) theta 
fcn_ddb <- function(theta) rep(1,length(theta)) 
fcn_dg <- function(mu) rep(1,length(mu)) 
pseudodata <- data # for initialization (no transformation) } 

return(pseudodata) 
}




V <- list() 

for (k in 1:K) { V[[k]] <- matrix(0, nrow = p[k], ncol = numBC) } 

for (r in 1:numBC) { prevtensor <- TensProd_GL(V, 1:(r-1)) # est of biclusters from prev ranks, as offset

# initialization
V_curr <- parafac_GL(pseudodata - prevtensor, 1)

#alternate between different directions
diff <- Inf 
niter <- 1 
rec_diff <- c() 

while (diff > Tol_outer && niter <= Niter_outer) { V_old <- V_curr

for (dir in 1:K) {
  # unfold data and prevtensor along dir
  X <- matrix(0, nrow = p[dir], ncol = prod(p) / p[dir])
  for (i in 1:prod(p) / p[dir]) {
    X[, i] <- unlist(data, use.names = FALSE)[which(rep(1:prod(p) / p[dir], each = p[dir]) == i)]
  }
  X <- t(X)
  offset <- matrix(0, nrow = p[dir], ncol = prod(p) / p[dir])
  for (i in 1:prod(p) / p[dir]) {
    offset[, i] <- unlist(prevtensor, use.names = FALSE)[which(rep(1:prod(p) / p[dir], each = p[dir]) == i)]
  }
  offset <- t(offset)
  v_fix <- 1 # Kron(vK,vK-1, ... v1)
  for (temp in Index[Index != dir]) {
    v_fix <- kronecker(V_curr[[temp]], v_fix) # length-prod(p)/p(dir)
  }
  
  # est loading dir, denoted by u
  niter_u <- 1
  diff_u <- Inf
  u <- V_curr[[dir]]
  while (diff_u > Tol_inner && niter_u < Niter_inner) { # IRLS+penalty (for normal, intrinsically no iteration)
    u_old <- u
    # calc weight matrix
    eta <- v_fix %*% u + offset # [prod(p)/p(dir)]*[p(dir)], vec()=true eta
    mu <- fcn_db(eta) # [prod(p)/p(dir)]*[p(dir)]
    sw <- 1 / sqrt(fcn_ddb(eta) * ((fcn_dg(mu)) ^ 2))
    Xmat <- sweep(sw * v_fix, 2, STATS = sw, FUN = `*`) # W^0.5*X but condensed
    Ymat <- sw * ((eta - offset) + (t(X) - mu) * fcn_dg(mu)) # W^0.5*Y but condensed
    
    # insert lasso regression here (Ymat~Xmat)
    u <- glmnet::glmnet(x = Xmat, y = Ymat, alpha = 1, lambda = 0)$beta # output is sparse loading u
    
    # update stopping rule
    niter_u <- niter_u + 1
    diff_u <- PrinAngle(u, u_old)
  }
  V_curr[[dir]] <- u
  
  # check if u is all zero
  if (sum(u == 0) == p[dir]) {
    print(sprintf("fail in dir %d", dir))
    stophere <- 1 # early termination sign
  }
}

# # update stopping rule
# diff <- 0
# for (k in 1:K) {
#   diff <- diff + PrinAngle(V_curr[[k]], V_old[[k]])
# }
# rec_diff <- c(rec_diff, diff)
# niter <- niter + 1
# }

# if (stophere == 1) { break }

# V[[r]] <- V_curr

# output bicluster rank r, for debugging
# if (fig == 1) { 
# 	Vdebug <- list() 
# 	for (k in 1:K) { Vdebug[[k]] <- matrix(0, nrow = p[k], ncol = r) 
# 		for (temp in 1:r


# check if early stop is needed
if (stophere) {
  break
}

# normalize (just the norm of loadings)
const <- 1
for (k in 2:K) {
  const0 <- norm(V_curr[[k]], type = "F")
  const <- const * const0
  V_curr[[k]] <- V_curr[[k]] / const0
}
V_curr[[1]] <- V_curr[[1]] * const # put all norm to the first loading

# stopping rule
diff <- 0
for (k in Index) {
  diff <- diff + PrinAngle(V_curr[[k]], V_old[[k]])
}
rec_diff <- c(rec_diff, diff)
niter <- niter + 1

# plot
if (fig) {
  plot(1:(niter-1), rec_diff, type = "l",
       main = "Sum of Angle diff for Loadings")
}

# check if r'th bicluster exists
if (stophere) {
  cat("The ", r, "th cocluster does NOT exist. Return with ", r-1, " clusters...\n")
  break
}

# check if r'th bicluster alternation converges
if (niter > Niter_outer) {
  cat(r, "th cocluster: NOT converge after ", Niter_outer, " iterations!\n")
} else {
  cat(r, "th cocluster: converge after ", niter-1, " iterations.\n")
}

for (k in Index) {
  V[[k]][, r] <- V_curr[[k]]
}


















PrinAngle <- function(V1, V2, paramstruct = list(ind = 1)) {
  # if ind = 1 (default), calc max principal angle
  # if ind = 2, Calculates All principal angles between column space of V1 and V2
  ind <- 1
  if (!is.null(paramstruct$ind)) {
    ind <- paramstruct$ind
  }
  
  p1 <- nrow(V1)
  r1 <- ncol(V1)
  p2 <- nrow(V2)
  r2 <- ncol(V2)
  if (p1 != p2) {
    stop("Input must be matched")
  }
  
  V1 <- svd(V1)$u
  V2 <- svd(V2)$u
  if (ind == 1) {
    angle <- 180 / pi * acos(min(svd(t(V1) %*% V2)$d))
  } else if (ind == 2) {
    angle <- 180 / pi * acos(svd(t(V1) %*% V2)$d)
  }
  
  return(angle)
}

mylasso <- function(Xmat, Ymat) {
  # This function provides explicit solution to lasso with orthogonal design
  # 1/2(y-X*beta)^2+lambda|beta|, where X'X is diagonal (but may not be I)
  # Moreover, our focused X is block diagonal (each block being a column in Xmat)
  # correspondingly, we condense y in a matrix form Ymat
  
  # OLS
  XtX <- diag(t(Xmat) %*% Xmat) # X'*X, diagonal matrix
  XtY <- diag(t(Xmat) %*% Ymat) # X'*y
  beta_OLS <- (1 / XtX) * XtY
  nn <- nrow(Xmat) * ncol(Xmat) # sample size in lasso regression
  
  lambda_cand <- (0:100) / 100 * max(abs(beta_OLS) * XtX)
  
  BIC_rec <- rep(0, 101)
  for (i in 1:101) {
    lambda <- lambda_cand[i]
    # lasso solution
    beta <- sign(beta_OLS) * pmax(abs(beta_OLS) - lambda / XtX, 0)
    
    # BIC
    sigma2 <- norm(Ymat - Xmat %*% beta, type = "F") ^ 2 / nn # SSE/sample size
    BIC_rec[i] <- log(sigma2) + sum(beta != 0) * log(nn) / nn
  }
  index <- which.min(BIC_rec)
  lambda <- lambda_cand[index]
  beta <- sign(beta_OLS) * pmax(abs(beta_OLS) - lambda / XtX, 0)
  
  # plot(lambda_cand, BIC_rec)
  
  return(list(beta = beta, lambda = lambda))
}






TensProd_GL <- function(L, range = 1:ncol(L[[1]])) {
  # Compute tensor product over multiple dimensions using Khatri-Rao product
  #
  # Input:
  #   L: List of length K, with matrix entries L[[k]]: m_k X R
  #   range: Column indices over which to apply tensor product. Default is all columns (range = 1:R)
  #
  # Output:
  #   Ans: Array of size m_1 X m_2 ... X m_K, where the [i1,...iK] entry is the sum of product L[[k]](i1,r)*...*L[[K]](iK,r) 
  #        over all r in range.
  
  K <- length(L)
  m <- rep(1, K)
  for (i in 1:K) {
    m[i] <- nrow(L[[i]])
  }
  
  if (max(range) > ncol(L[[1]])) {
    stop("Range exceeds rank!")
  }
  
  # customize L
  newL <- L
  for (i in 1:K) {
    newL[[i]] <- L[[i]][, range]
  }
  L <- newL
  
  tempL <- L[K:2]
  matX <- L[[1]] %*% t(kronecker(tempL)) # X_(1)
  Ans <- array(as.vector(matX), dim = m)
  
  return(Ans)
}



# %Performs parafac factorization via ALS
# %
# % Input
# %       Y           Array of dimension m1 X m2 X ... X mK 
# %       R           Desired rank of the factorization  
# %                   Default is all columns (range = [1:R]).
# %
# % Output
# %       U           List of basis vectors for parafac factorization, 
# %                   U{k}: mk X R  for k=1,...,K. Columns of U{K} have norm
# %                   1 for k>1.  TensProd_GL(U) approximates Y
# %      
# %       SqError     Vector of squared error for the approximation at each
# %                   iteration (should be non-increasing).  
# %
# % Created: 07/10/2016
# % By: Eric F. Lock
# %
# % Modified by Gen Li: 8/17/2016
# %   1. change TensProd to TensProd_GL to save time
# % %




parafac_GL <- function(Y, R) {
  m <- dim(Y)
  L <- length(m)
  U <- list()
  for (l in 2:L) {
    U[[l]] <- sweep(matrix(rnorm(m[l] * R), ncol = R), MARGIN = 2, STATS = colSums(U[[l]]^2)^0.5, FUN = '/')
  }
  Index <- 1:L
  
  SqError <- numeric()
  i <- 0
  thresh <- 10^(-1)
  SqErrorDiff <- thresh + 1
  Yest <- array(0, dim = m)
  iters <- 1000
  while (i < iters && SqErrorDiff > thresh) {
    i <- i + 1
    Yest_old <- Yest
    for (l in 1:L) {
      ResponseMat <- matrix(permute(Y, c(Index[Index != l], l)), nrow = prod(m[Index[Index != l]]), ncol = m[l])
      PredMat <- matrix(0, nrow = prod(m[Index[Index != l]]), ncol = R)
      for (r in 1:R) {
        Temp <- TensProd_GL(U[Index != l], r)
        PredMat[, r] <- matrix(Temp, nrow = prod(m[Index[Index != l]]))
      }
      U[[l]] <- solve(t(PredMat) %*% PredMat) %*% t(PredMat) %*% ResponseMat
      if (l > 1) U[[l]] <- sweep(U[[l]], MARGIN = 2, STATS = colSums(U[[l]]^2)^0.5, FUN = '/')
    }
    Yest <- TensProd_GL(U)
    SqError[i] <- norm(Y - Yest, "F")^2
    if (i > 1) SqErrorDiff <- abs(SqError[i] - SqError[i - 1])
  }
  list(U = U, SqError = SqError)
}