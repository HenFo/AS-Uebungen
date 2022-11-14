library(purrr)

find.zero <- function(start.vec, grads, precision = 1e-10, max.iter = 20) {
  xk <- start.vec
  xkp <- start.vec
  delta <- Inf
  k <- 0
  
  while(delta > precision && k < max.iter) {
    grad.calc <- do.call(grads, as.list(xkp))
    grad <- t(attr(grad.calc, "grad"))
    hesse.inv <- solve(matrix(attr(grad.calc, "hessian"), nrow = length(start.vec)))
    
    xkp <- xk - hesse.inv %*% grad
    delta <- norm(xk - xkp)
    xk <- xkp
    k <- k+1
  }
  
  
  return(xkp)
}



check.hesse <- function(zero.vec, grads) {
  grad.calc <- do.call(grads, as.list(zero.vec, ))
  hesse <- matrix(attr(grad.calc, "hessian"), nrow = length(zero.vec))
  
  section.det <- list()
  for (i in seq_len(nrow(hesse))) {
    section <- round(as.matrix(hesse[1:i, 1:i]), digits = 2)
    section.det[[i]] <- det(section)
  }
  
  definit.pos <- all(section.det > 0)
  definit.neg <- all(section.det[seq(1, length(section.det), 2)] < 0) &&
    all(section.det[seq(2, length(section.det), 2)] > 0)
  
  if (definit.pos)
    return("pos-def")
  else if(definit.neg)
    return("neg-def")
  else
    return("indef")
  
}

##########################
lower_bound <- 0
upper_bound <- 2*pi
f <- expression(sin(x)*sin(y))
vars <- c("x", "y")
##########################


random_points <- runif(10, lower_bound, upper_bound)
start.vecs <- as.matrix(expand.grid(random_points, random_points))
start.vecs <- unname(start.vecs)

f.gradients <- deriv3(f, vars, function.arg = T)

f.zero <- apply(start.vecs, 1,
                 function(x) find.zero(x, grads = f.gradients, max.iter = 10))
f.zero <- unique(t(f.zero))
f.zero <- f.zero[apply(f.zero <= upper_bound & f.zero >= lower_bound, 1, all),]

max_min <- apply(f.zero,1, function(x) check.hesse(x, f.gradients))

result <- cbind(f.zero, max_min)
result
