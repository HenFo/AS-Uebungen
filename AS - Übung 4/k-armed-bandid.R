# install.packages("tidyverse")
library(tidyverse)

eps_greedy <- function(means, epsilon, timesteps) {
	# Erhaltene Rewards
	rewards <- vector("numeric", timesteps)

	# Herkunft des erhaltenen Rewards
	by <- vector("integer", timesteps)

	# Zufällige Wahl der ersten Aktion
	first_action <- sample(seq_along(means), 1)
	rewards[1] <- rnorm(1, means[first_action])
	by[1] <- first_action

	for (i in seq_len(timesteps - 1)) {
		# Mittlerer Reward pro Herkunft.
		# Wurde eine Aktion noch nicht gewählt, wird ein
		#	mittlerer Reward von 0 angenommen
		current_mean_rewards <-
			tapply(rewards[1:i],
				   factor(by[1:i], levels = seq_along(means)),
				   mean,
				   default = 0)

		# Wählen der Aktion At
		# Finden der aktuellen reward-maximierenden Aktion
		max_action <- as.integer(names(which.max(current_mean_rewards)))
		# In epsilon % der Fälle wird eine zufällige Aktion gewählt
		action <-
			if (epsilon < runif(1))
				max_action
			else
				sample(seq_along(means), 1)

		# Speichern des neuen Rewards der gewählten Aktion
		# Rt mit Mittelwert q*(At)
		rewards[i + 1] <- rnorm(1, means[action])
		by[i + 1] <- action
	}

	# Ausgabe als Data Frame
	tibble(timestep = seq_len(timesteps),
		   reward = rewards,
		   by = by)
}


k_armed_bandid <- function(k, epsilon, timesteps, runs) {
	# Funktion zum vordefinieren von Parametern.
	# means wird über das Mapping in die Funktion gegeben
	map_builder <- function(epsilon, timesteps) {
		function (means) {
			eps_greedy(means, epsilon, timesteps)
		}
	}

	# Erzeugen von runs Verteilungen q*(a)
	repetitions <- tibble(run = seq_len(runs),
						  means = replicate(runs, rnorm(k), simplify = F))

	# Ausführen der einzelnen Runs und anschließendem
	#	Mitteln der Rewards pro Timestep
	repetitions <- repetitions |>
		mutate(rewards = map(
			means,
			map_builder(epsilon, timesteps),
			.progress = list(clear = F)
		)) |>
		select(rewards) |>
		unnest(rewards) |>
		group_by(timestep) |>
		summarise(mean_reward = mean(reward))

	repetitions
}


# Definieren der Parameterkombinationen
# expand_grid erzeugt jede mögliche Parameterkombination
params <- expand_grid(
	k = c(5, 10, 20),
	epsilon = c(0, 0.01, 0.1, 0.25),
	timesteps = c(1000),
	runs = c(500, 1000)
)

# Ausführen des K-Armed-Bandid mit allen Parameterkombinationen
all_mean_rewards <- params %>%
	mutate(mean_rewards = pmap(
		.,
		k_armed_bandid,
		.progress = list(name = "Parameterkombination",
						 clear = F)
	))

save(all_mean_rewards, file = "all_mean_rewards.RData")

params <- expand_grid(
	k = c(10),
	epsilon = c(0, 0.01, 0.1, 0.25),
	timesteps = c(2000),
	runs = c(500)
)

all_mean_rewards_2000 <- params %>%
	mutate(mean_rewards = pmap(
		.,
		k_armed_bandid,
		.progress = list(name = "Parameterkombination",
						 clear = F)
	))

save(all_mean_rewards, file = "all_mean_rewards_2000.RData")

