package experiments

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/victorddiniz/fitness-function-analysis/es"
	"github.com/victorddiniz/fitness-function-analysis/functions"
	"github.com/victorddiniz/fitness-function-analysis/iohandlers"
)

// Experiment ...
type Experiment struct {
	rand           *rand.Rand
	numTests       int
	mu             int
	lambda         int
	maxIteractions int
	datasetPath    string
	fitFunctions   []func(t, o []float64) float64
	errorMeasures  []func(t, o []float64) float64
}

// Run ...
func (e *Experiment) Run() ([][]float64, [][]float64) {
	results := make([][]float64, len(e.fitFunctions))
	predictions := make([][]float64, len(e.fitFunctions))

	for i, fitFunc := range e.fitFunctions {
		results[i] = make([]float64, len(e.errorMeasures)+2)
		fitValues := make([]float64, e.numTests)
		var bestInd *es.Individual
		bestValidation := 0.0

		for j := 0; j < e.numTests; j++ {
			ee, err := es.NewEvolutionExporter(fmt.Sprintf("evolution-%d.csv", j))
			if err != nil {
				panic(err)
			}
			population := es.NewPopulation(e.mu, e.lambda, e.maxIteractions, e.datasetPath, e.rand, fitFunc, ee)
			ioHandler := iohandlers.GetInstance()

			indRun, _, _ := population.Run()
			in, out := ioHandler.GetKLagValidationSet(indRun.GetLag())
			value := indRun.Fitness(in, out)
			fitValues[j] = value

			if bestInd == nil || value > bestValidation {
				bestInd = indRun
				bestValidation = value
			}
		}

		//calculate mean and std
		sumFit := 0.0
		for _, value := range fitValues {
			sumFit += value
		}
		meanFit := sumFit / float64(e.numTests)
		sumDiffSquared := 0.0
		for _, value := range fitValues {
			sumDiffSquared += (value - meanFit) * (value - meanFit)
		}
		stdFit := math.Sqrt(sumDiffSquared / float64(e.numTests))
		results[i][len(e.errorMeasures)+1] = stdFit
		results[i][len(e.errorMeasures)] = meanFit

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
func NewExperiment(randSeed int64, numTests, mu, lambda, maxIteractions int, datasetPath string) *Experiment {
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

	errorMeasures := []func(t, o []float64) float64{
		functions.ARV,
		functions.MAPE,
		functions.MSE,
		functions.POCID,
		functions.Theil,
	}

	return &Experiment{
		rand:           rand,
		numTests:       numTests,
		mu:             mu,
		lambda:         lambda,
		maxIteractions: maxIteractions,
		datasetPath:    datasetPath,
		fitFunctions:   fitFunctions,
		errorMeasures:  errorMeasures,
	}
}
