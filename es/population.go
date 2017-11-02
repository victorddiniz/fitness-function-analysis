package es

import (
	"sort"
	"math/rand"
	"time"
	"github.com/victorddiniz/fitness_function_analysis/iohandlers"
)

// Population ...
type Population struct {
	pop           []*Individual
	mu            int
	lambda        int
	iteration     int
	maxIterations int
	fitness       map[*Individual]float64
	lastFitnessValidation float64
}

func (population *Population) Len() int { return len(population.pop) }
func (population *Population) Swap(i, j int) {
	population.pop[i], population.pop[j] = population.pop[j], population.pop[i]
}
func (population *Population) Less(i, j int) bool {
	return population.f(population.pop[i]) > population.f(population.pop[j])
}

func (population *Population) initPop() {
	for _, ind := range population.pop {
		population.f(ind)
	}
}

func (population *Population) parentReplacement() {
	for index, ind := range population.pop {
		bestChild := ind

		for i := 0; i < population.lambda; i++ {
			child := ind.Mutate()
			if population.f(child) >= population.f(bestChild) {
				population.pop[index] = child
				delete(population.fitness, bestChild)
				bestChild = child
			}
		}
	}
}

func (population *Population) f(individual *Individual) float64 {
	value, ok := population.fitness[individual]
	if !ok {
		ioHandler := iohandlers.GetInstance()
		in, out := ioHandler.GetKLagTrainingSet(individual.lag)
		value = individual.Fitness(in, out)
		population.fitness[individual] = value
	}
	return value
}

func (population *Population) hasReachedLimit() bool {
	bestInd := population.pop[0]
	ioHandler := iohandlers.GetInstance()
	in, out := ioHandler.GetKLagValidationSet(bestInd.GetLag())
	fitnessValidation := bestInd.Fitness(in, out)
	rateDeacrease := fitnessValidation/population.lastFitnessValidation

	hasReachedEnd := population.iteration > population.maxIterations ||
	(population.iteration > 1 && rateDeacrease <= 0.99)

	population.lastFitnessValidation = fitnessValidation
	return hasReachedEnd
}

// Run ...
func (population *Population) Run() (*Individual, float64) {
	population.initPop()
	population.iteration = 0
	//var bestInd * Individual

	for {
		sort.Sort(population)
		population.parentReplacement()
		population.iteration++
		if !population.hasReachedLimit() { break }
	}
	sort.Sort(population)

	return population.pop[0], population.f(population.pop[0])
}

// NewPopulation ...
func NewPopulation(mu, lambda, maxInterations int, datasetPath string) *Population {
	pop := make([]*Individual, mu)
	seed := rand.NewSource(time.Now().UnixNano())
	randSource := rand.New(seed)

	for index := range pop {
		lag := randSource.Intn(10) + 1
		hidden := randSource.Intn(10) + 1
		pop[index] = NewIndividual(lag, hidden)
	}

	iohandlers.NewIOHandler(datasetPath, 10)

	population := &Population{
		pop:           pop,
		mu:            mu,
		lambda:        lambda,
		maxIterations: maxInterations}
	population.fitness = make(map[*Individual]float64)
	population.lastFitnessValidation = -1.0

	return population
}
