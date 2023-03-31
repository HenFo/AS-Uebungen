from ast import main
import os
from abc import ABC, abstractclassmethod
from functools import total_ordering
from multiprocessing import Pool, freeze_support
from typing import Any, Callable

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class Gene(ABC):
	def __init__(self, allele:Any, mutation_rate:float) -> None:
		super().__init__()
		self.allele = allele
		self.mutation_rate = mutation_rate

	@abstractclassmethod
	def mutate(self) -> "Gene":
		raise NotImplementedError

	def __str__(self) -> str:
		return str(self.allele)

	def __repr__(self) -> str:
		return f"Gene({str(self.allele)})"


class FloatGene(Gene):
	def __init__(self, allele:float = None, mutation_rate:float = 0.2, range:tuple = None, mutation_sd:float = 1) -> None:
		if range is None:
			allele = np.random.normal() if allele is None else allele
		else:
			allele = np.random.uniform(range[0], range[1]) if allele is None else allele

		super().__init__(allele, mutation_rate)
		self.range = (-np.inf, np.inf) if range is None else range
		self.mutation_sd = mutation_sd
	
	def mutate(self) -> "FloatGene":
		if np.random.random() <= self.mutation_rate:
			mut_allele = self.allele + np.random.normal(scale=self.mutation_sd)
			mut_allele = mut_allele if mut_allele > self.range[0] else self.range[0]
			mut_allele = mut_allele if mut_allele < self.range[1] else self.range[1]
			return FloatGene(mut_allele, self.mutation_rate, self.range, self.mutation_sd)
		return self
	

class BinaryGene(Gene):
	def __init__(self, allele:int = None, mutation_rate:float = 0.2) -> None:
		allele = allele if allele is not None else 0 if np.random.random() < 0.5 else 1
		super().__init__(allele, mutation_rate)

	def mutate(self) -> "BinaryGene":
		mut_allele = self.allele if np.random.random() > self.mutation_rate else (self.allele + 1) % 2
		return BinaryGene(mut_allele, self.mutation_rate)


class IntGene(Gene):
	def __init__(self, allele: int = None, mutation_rate:float = 0.2, range:tuple = None, mutation_span:float = (-5,5)) -> None:
		if range is None:
			allele = np.random.randint(0, 100) if allele is None else allele
		else:
			allele = np.random.randint(range[0], range[1]) if allele is None else allele

		super().__init__(allele, mutation_rate)
		self.range = range
		self.mutation_span = mutation_span
	
	def mutate(self) -> "IntGene":
		mut_allele = self.allele if np.random.random() > self.mutation_rate \
			else self.allele + np.random.randint(self.mutation_span[0], self.mutation_span[1])
		mut_allele = mut_allele if mut_allele < self.range[1] else self.range[1]
		mut_allele = mut_allele if mut_allele > self.range[0] else self.range[0]

		return IntGene(mut_allele, self.mutation_rate, self.range, self.mutation_span)



@total_ordering
class Chromosome(list):
	def __init__(self, genes:list[Gene]) -> None:
		assert all(map(lambda x: issubclass(type(x), Gene), genes)), "Invalid types. Should all be of type 'Gene'"
		super().__init__(genes)
		self.fitness_value:float = np.inf

	def mutate(self) -> "Chromosome":
		return Chromosome([x.mutate() for x in self])

	def as_tuple(self) -> tuple:
		return tuple(x.allele for x in self)

	def set_fitness(self, fitness_value:float) -> None:
		self.fitness_value = fitness_value

	@classmethod
	def create_offsprings(self, p1:"Chromosome", p2:"Chromosome", split_points:list[int] = None) -> tuple["Chromosome"]:
		assert len(p1) == len(p2), "Chromosomes must be same length"
		assert max(split_points) < len(p1), f"Split point {max(split_points)} too large for Chromosome of length {len(p1)}"
		split_points = split_points if split_points is not None else [int(len(p1)/2)]

		def _split_chromosome(l:list, s:list) -> list[list]:
			start = 0
			splits = []
			for end in s:
				splits.append(l[start:end])
				start = end
			splits.append(l[start:])
			return splits

		def _join_chromosomes(l1:list[list], l2:list[list]) -> tuple["Chromosome", "Chromosome"]:
			a, b = [], []
			for i in range(0, len(l1), 2):
				a += l1[i]
				b += l2[i]
				if i+1 < len(l1):
					a += l2[i+1]
					b += l1[i+1]
			return Chromosome(a), Chromosome(b)
		
		# Split chromosomes along the split points
		c_split1, c_split2 = _split_chromosome(p1, split_points), _split_chromosome(p2, split_points)
		off1, off2 = _join_chromosomes(c_split1, c_split2)

		return (off1, off2)
	
	def __repr__(self) -> str:
		return f"Chromo({self.fitness_value}, {super().__repr__()})"

	def __lt__(self, other:"Chromosome") -> bool:
		return self.fitness_value < other.fitness_value
	
	def __eq__(self, other:"Chromosome") -> bool:
		return self.fitness_value == other.fitness_value
	
	def __hash__(self):
		return hash(self.as_tuple())

	def __add__(self, other:"Chromosome") -> float:
		return self.fitness_value + other.fitness_value


class Population(list):
	def __init__(self, chromosomes: list[Chromosome], fitness_func:Callable = None) -> None:
		assert all(map(lambda x: type(x) == Chromosome, chromosomes)), "Invalid types. Should all be of type 'Gene'"
		assert all([len(lst) == len(chromosomes[0]) for lst in chromosomes]), "Chromosomes don't have the same length"
		super().__init__(chromosomes)

		self.fitness_func = fitness_func
		n = len(self)
		a = np.arange(n)[::-1]
		self.rank_probabilities = np.array([(i+1)/(n*(n+1)/2) for i in a])


	def evolve(self, strategy:str = "2,2", max_iter:int = 100, precision:float = 1e-2, patience:int = 10, parallel:bool = False) -> "Population":
		assert strategy in ("2+2", "2,2")
		if strategy == "2+2":
			return self._evolve_parent_or_child(max_iter, precision, patience, parallel)
		if strategy == "2,2":
			return self._evolve_children_only(max_iter, precision, parallel)
	
	def _evolve_population(sorted_pop:"Population") -> "Population":
			offsprings:list[Chromosome] = list()
			print("current best error =", sorted_pop[0].fitness_value)
			while len(offsprings) < len(sorted_pop):
				p1, p2 = np.random.choice(len(sorted_pop), size=2, p=sorted_pop.rank_probabilities)
				p1 = sorted_pop[p1]
				p2 = sorted_pop[p2]
				off1, off2 = Chromosome.create_offsprings(p1,p2)
				offsprings.append(off1.mutate())
				offsprings.append(off2.mutate())

			return Population(offsprings, sorted_pop.fitness_func)


	def _evolve_children_only(self, max_iter:int = 100, precision:float = 1e-2, parallel:bool = False) -> "Population":
		current_pop = self
		history = {}
		for _ in range(max_iter):
			current_pop.calc_fitness(parallel)
			for i, chrom in enumerate(sorted(current_pop)):
				history[i] = history.get(i, []) + [chrom.fitness_value]
			current_pop = self._evolve_population(current_pop)
		
		return history, current_pop


	def _evolve_parent_or_child(self, max_iter:int = 100, precision:float = 1e-2, patience:int = 10, parallel:bool = False) -> "Population":
		current_pop = self
		history = {}
		for _ in range(max_iter):
			current_pop.calc_fitness(parallel)

			for i, chrom in enumerate(sorted(current_pop)):
				history[i] = history.get(i, []) + [chrom.fitness_value]
			evolved_pop = self._evolve_population(current_pop)
			current_pop = sorted(current_pop + evolved_pop)
		
		return history, current_pop

	def calc_fitness(self, parallel:bool = False) -> None:
		assert self.fitness_func is not None
		if not parallel:
			for x in tqdm(self, position=0, desc="Population", leave=True):
				fitness = self.fitness_func(x)
				x.set_fitness(fitness)
		else:
			fitness_vals = process_map(self.fitness_func, self, max_workers=min(32, os.cpu_count() + 4, len(self)))
			for x, f in zip(self, fitness_vals):
				x.set_fitness(f)
		

	def __repr__(self) -> str:
		return "Population(\n\t" + "\n\t".join([repr(x) for x in self]) + "\n)"


def fitness_function(func:Callable) -> Callable:
	def filter_calculated(chrom:Chromosome) -> float:
		if chrom.fitness_value < np.inf:
			return chrom.fitness_value
		return func(chrom)
	return filter_calculated


if __name__ == '__main__':

	@fitness_function
	def loss(c:Chromosome) -> float:
		return c[0].allele

	a = Chromosome([FloatGene()])
	a.set_fitness(5)
	b = Chromosome([FloatGene()])

	print(loss(a))
	print(loss(b))