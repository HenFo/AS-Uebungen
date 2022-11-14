library(tidyverse)
library(lattice)

create.offspring <- function(parents, popsize, mean, std) {
  offsprings <- tibble(
    x = rnorm(popsize, mean, std) + parents$x,
    y = rnorm(popsize, mean, std) + parents$y,
    fitness = f(x,y)
  )
  
  return(offsprings)
}

plot_function2d <- function(f, resolution, lower_bound, upper_bound) {
  x <- seq(lower_bound, upper_bound, length.out = resolution)
  g <- expand.grid(x=x,y=x)
  g$z <- f(g$x, g$y)
  
  wireframe(z ~ x * y, 
            data=g,
            scales = list(arrows = FALSE),
            drape = TRUE,
            col.regions=rainbow(100))
}

f <- function(x,y) (x^2 + y^2)

k <- 200
popsize <- 10
lower_bound <- -2*pi*2
upper_bound <- 2*pi*2

plot_function2d(f, 20, lower_bound, upper_bound)

population <- tibble(
  x = runif(popsize, lower_bound, upper_bound),
  y = runif(popsize, lower_bound, upper_bound),
  fitness = f(x,y)
)

while(k > 0) {
  offspring <- create.offspring(population, popsize, 0, 1)
  population <- bind_rows(population, offspring) %>%
    arrange(fitness) %>%
    head(popsize)
  k <- k-1
}
population



