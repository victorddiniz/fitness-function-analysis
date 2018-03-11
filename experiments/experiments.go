package experiments

import (
	"github.com/victorddiniz/fitness-function-analysis/iohandlers"
	"github.com/victorddiniz/fitness-function-analysis/es"
	"github.com/victorddiniz/fitness-function-analysis/functions"
	"math/rand"
)

// Experiment ...
type Experiment struct {
	rand *rand.Rand
	numTests int
	mu int
	lambda int
	maxIteractions int
	datasetPath string
	fitFunctions []func(t, o []float64) float64
	errorMeasures []func(t, o []float64) float64
}

// Run ...
func (e *Experiment) Run() ([][]float64, [][]float64){
	results := make([][]float64, len(e.fitFunctions))
	predictions := make([][]float64, len(e.fitFunctions))

	for i, fitFunc := range e.fitFunctions {
		results[i] = make([]float64, len(e.errorMeasures))
		var bestInd *es.Individual
		bestValidation := 0.0

		for j := 0; j < e.numTests; j++ {
			population := es.NewPopulation(e.mu, e.lambda, e.maxIteractions, e.datasetPath, e.rand, fitFunc)
			ioHandler := iohandlers.GetInstance()
			
			indRun, _, _ := population.Run()
			in, out := ioHandler.GetKLagValidationSet(indRun.GetLag())
			value := indRun.Fitness(in, out)

			if bestInd == nil || value > bestValidation {
				bestInd = indRun
				bestValidation = value
			}
		}

		ioHandler := iohandlers.GetInstance()
		in, target := ioHandler.GetKLagTestSet(bestInd.GetLag())
		obs := bestInd.Predict(in)
		predictions[i] = obs

		for j, errorFunc := range e.errorMeasures {
			results[i][j] = errorFunc(target, obs)
		}
	}

	return results, predictions
}

// NewExperiment ...
func NewExperiment(randSeed int64, numTests, mu, lambda, maxIteractions int, datasetPath string) *Experiment{
	rand := rand.New(rand.NewSource(randSeed))

	fitFunctions := []func(t, o []float64) float64{
		functions.F1,
		functions.F2,	
		functions.F3,
		functions.F4,
		functions.F5,
		functions.F6,
		functions.F7,
		functions.F8,
	}

	errorMeasures := []func(t, o []float64) float64 {
		functions.ARV,
		functions.MAPE,
		functions.MSE,
		functions.POCID,
		functions.Theil,
	}

	return &Experiment{
		rand: rand,
		numTests: numTests,
		mu: mu,
		lambda: lambda,
		maxIteractions: maxIteractions,
		datasetPath: datasetPath,
		fitFunctions: fitFunctions,
		errorMeasures: errorMeasures,
	}
}