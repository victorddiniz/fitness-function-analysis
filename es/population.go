package es

import (
	"math/rand"
	"sort"

	"github.com/victorddiniz/fitness-function-analysis/iohandlers"
)

// Population ...
type Population struct {
	pop                   []*Individual
	mu                    int
	lambda                int
	iteration             int
	maxIterations         int
	fitness               map[*Individual]float64
	lastFitnessValidation float64
	ee                    *EvolutionExporter
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
	//fitnessValidation := bestInd.Fitness(in, out)
	//rateDeacrease := fitnessValidation / population.lastFitnessValidation

	hasReachedEnd := population.iteration > population.maxIterations ||
		false //(population.iteration > 1 && rateDeacrease <= 0.99)

	//population.lastFitnessValidation = fitnessValidation

	if population.iteration%20 == 0 || population.iteration == 1 {
		target := out
		observations := bestInd.Predict(in)
		population.ee.WriteEvolution(target, observations, population.fitness[bestInd])
	}

	return hasReachedEnd
}

// Run ...
func (population *Population) Run() (*Individual, float64, int) {
	population.initPop()
	population.iteration = 0
	//var bestInd * Individual

	sort.Sort(population)
	for {
		population.parentReplacement()
		population.iteration++
		sort.Sort(population)
		if population.hasReachedLimit() {
			break
		}
	}

	population.ee.Close()
	return population.pop[0], population.f(population.pop[0]), population.iteration
}

// NewPopulation ...
func NewPopulation(mu, lambda, maxInterations int, datasetPath string, randGen *rand.Rand, fitFunction func(t, o []float64) float64, ee *EvolutionExporter) *Population {
	pop := make([]*Individual, mu)
	maxLag := 20
	maxHidden := 30

	for index := range pop {
		lag := randGen.Intn(maxLag) + 1
		hidden := randGen.Intn(maxHidden) + 1
		pop[index] = NewIndividual(lag, hidden, randGen, fitFunction)
	}

	iohandlers.NewIOHandler(datasetPath, maxLag)

	population := &Population{
		ee:                    ee,
		pop:                   pop,
		mu:                    mu,
		lambda:                lambda,
		maxIterations:         maxInterations,
		fitness:               make(map[*Individual]float64),
		lastFitnessValidation: -1.0,
	}

	return population
}
