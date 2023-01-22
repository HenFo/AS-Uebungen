from abc import ABC, abstractclassmethod
from functools import total_ordering
from typing import Any, Callable
import numpy as np


class Gene(ABC):
	def __init__(self, allele:Any) -> None:
		super().__init__()
		self.allele = allele

	@abstractclassmethod
	def mutate(self) -> "Gene":
		raise NotImplementedError

	def __str__(self) -> str:
		return str(self.allele)

	def __repr__(self) -> str:
		return f"Gene({str(self.allele)})"


class FloatGene(Gene):
	def __init__(self, allele:float = None, mutation_rate:float = 0.2, range:tuple = (-np.inf, np.inf), mutation_sd:float = 1) -> None:
		allele = np.random.normal() if allele is None else allele
		super().__init__(allele)
		self.range = range
		self.mutation_rate = mutation_rate
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
		super().__init__(allele)
		self.mutation_rate = mutation_rate

	def mutate(self) -> "BinaryGene":
		mut_allele = self.allele if np.random.random() > self.mutation_rate else (self.allele + 1) % 2
		return BinaryGene(mut_allele, self.mutation_rate)




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
	def create_offspring(self, c1:"Chromosome", c2:"Chromosome") -> tuple["Chromosome", "Chromosome"]:
		assert len(c1) == len(c2)
		middle = int(len(c1) / 2)
		off1 = Chromosome(c1[:middle] + c2[middle:])
		off2 = Chromosome(c2[:middle] + c1[middle:])

		return (off1, off2)
	
	def __repr__(self) -> str:
		return f"Chromo({self.fitness_value}, {super().__repr__()})"

	def __lt__(self, other:"Chromosome") -> bool:
		return self.fitness_value < other.fitness_value
	
	def __eq__(self, other:"Chromosome") -> bool:
		return self.fitness_value == other.fitness_value
	
	def __hash__(self):
		return hash(self.as_tuple())


class Population(list):
	def __init__(self, chromosomes: list[Chromosome], fitness_func:Callable = None) -> None:
		assert all(map(lambda x: type(x) == Chromosome, chromosomes)), "Invalid types. Should all be of type 'Gene'"
		assert all([len(lst) == len(chromosomes[0]) for lst in chromosomes]), "Chromosomes don't have the same length"
		super().__init__(chromosomes)

		self.fitness_func = fitness_func
		n = len(self)
		a = np.arange(n)[::-1]
		self.rank_probabilities = np.array([(i+1)/(n*(n+1)/2) for i in a])


	def evolve(self, strategy:str = "2+2", max_iter:int = 100, precision:float = 1e-2, patience:int = 10) -> "Population":
		assert strategy in ("2+2", "2,2")
		if strategy == "2+2":
			return self._evolve_parent_or_child(max_iter, precision, patience)
		if strategy == "2,2":
			return self._evolve_children_only(max_iter, precision)
	
	
	def _evolve_children_only(self, max_iter:int = 100, precision:float = 1e-2) -> "Population":
		def evolve_population(pop:"Population") -> "Population":
			pop.calc_fitness()
			offsprings:list[Chromosome] = list()
			sorted_pop = sorted(pop)
			while len(offsprings) < len(pop):
				p1, p2 = np.random.choice(len(pop), size=2, p=pop.rank_probabilities)
				p1 = sorted_pop[p1]
				p2 = sorted_pop[p2]
				off1, off2 = Chromosome.create_offspring(p1,p2)
				offsprings.append(off1.mutate())
				offsprings.append(off2.mutate())

			return Population(offsprings, pop.fitness_func)

		current_pop = self
		for i in range(max_iter):
			current_pop = evolve_population(current_pop)
			
		return current_pop



	def _evolve_parent_or_child(self, max_iter:int = 100, precision:float = 1e-2, patience:int = 10) -> "Population":
		pass

	def calc_fitness(self) -> None:
		assert self.fitness_func is not None
		for x in self:
			fitness = self.fitness_func(x)
			x.set_fitness(fitness)
		

	def __repr__(self) -> str:
		return "Population(\n\t" + "\n\t".join([repr(x) for x in self]) + "\n)"



def test_polynom():
	chromos = [Chromosome([FloatGene() for _ in range(2)]) for _ in range(10)]

	def fitness(chromo:Chromosome) -> float:
		x, y = chromo.as_tuple()
		return (x+2)**2 + (y-2) ** 2

	pop = Population(chromos, fitness)
	print(pop)
	fitted = pop.evolve("2,2")
	fitted.calc_fitness()
	print(Population(sorted(fitted)))


def rucksack_problem():
	rucksack = {
		"kette": {
			"value": 4,
			"weight": 3
		},
		"gold": {
			"value": 7,
			"weight": 7
		},
		"krone": {
			"value": 5,
			"weight": 4
		},
		"muenze": {
			"value": 1,
			"weight": 1
		},
		"beil": {
			"value": 4,
			"weight": 5
		},
		"schwert": {
			"value": 3,
			"weight": 4
		},
		"ring": {
			"value": 5,
			"weight": 2
		},
		"kelch": {
			"value": 1,
			"weight": 3
		}
	}
	kapa = 9

	def fitness(chromo:Chromosome, print_content:bool = False) -> float:
		chromo = chromo.as_tuple()
		lookup = list(rucksack.items())
		weight = sum([lookup[i][1]["weight"] for i, included in enumerate(chromo) if included == 1])
		value = sum([lookup[i][1]["value"] for i, included in enumerate(chromo) if included == 1]) 
		if weight > kapa:
			return np.inf
		
		if print_content:
			print(weight, value)
		return 1/(value + 1e-5)


	chromos = [Chromosome([BinaryGene() for _ in range(len(rucksack))]) for _ in range(20)]
	pop = Population(chromos, fitness)
	print(pop)
	fitted = pop.evolve("2,2")
	fitted.calc_fitness()
	fitted = Population(sorted(fitted))
	print(fitted)
	[fitness(x, True) for x in fitted]

rucksack_problem()