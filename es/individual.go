package es

import (
	"github.com/victorddiniz/fitness-function-analysis/functions"
	"math"
	"math/rand"
)

// Individual ...
type Individual struct {
	weights [][]float64
	mSteps [][]float64
	biasWeights []float64
	biasMSteps []float64
	t float64
	tLine float64
	hidden int
	lag int
	fitnessFunction func(t, o []float64) float64
	randGen *rand.Rand
}

func (ind *Individual) epsilon() float64 {
	return 1e-3
}

func (ind *Individual) mutateMS(i, j int) {
	var ms float64

	if j == -1 {
		ms = ind.biasMSteps[i]
	} else {
		ms = ind.mSteps[i][j]
	}

	t := ind.t
	tLine := ind.tLine

	ms *= math.Exp(tLine * ind.randGen.NormFloat64() + t * ind.randGen.NormFloat64())
	if ms < ind.epsilon() {
		ms = ind.epsilon()
	}

	if j == -1 {
		ind.biasMSteps[i] = ms
	} else {
		ind.mSteps[i][j] = ms
	}
}

func (ind *Individual) mutateWeight(i, j int) {
	var ms, weight float64

	if j == -1 {
		ms = ind.biasMSteps[i]
		weight = ind.biasWeights[i]
		ind.biasWeights[i] = weight + ms * ind.randGen.NormFloat64()
	} else {
		ms = ind.mSteps[i][j]
		weight = ind.weights[i][j]
		ind.weights[i][j] = weight + ms * ind.randGen.NormFloat64()
	}
}

func (ind *Individual) predictSolo(input []float64) float64 {
	intermediarySumSig := 0.0
	for i := 0; i < ind.hidden; i++ {
		partial := -ind.biasWeights[i] * float64(ind.hidden)
		for j := 0; j < ind.lag; j++ {
			partial += ind.weights[j][i] * input[j]
		}
		partial = functions.Sigmoid(partial)
		intermediarySumSig += partial * ind.weights[ind.lag][i]
	}
	return intermediarySumSig - ind.biasWeights[ind.hidden]

}

func (ind *Individual) copy() *Individual {
	var newInd Individual
	newInd.weights = make([][]float64, len(ind.weights))
	newInd.mSteps = make([][]float64, len(ind.mSteps))
	newInd.biasMSteps = make([]float64, len(ind.biasMSteps))
	newInd.biasWeights = make([]float64, len(ind.biasWeights))
	for i := 0; i <= ind.lag; i++ {
		newInd.weights[i] = make([]float64, ind.hidden)
		newInd.mSteps[i] = make([]float64, ind.hidden)
		copy(newInd.weights[i], ind.weights[i])
		copy(newInd.mSteps[i], ind.mSteps[i])
	}
	copy(newInd.biasMSteps, ind.biasMSteps)
	copy(newInd.biasWeights, ind.biasWeights)
	newInd.t, newInd.tLine = ind.t, ind.tLine
	newInd.hidden, newInd.lag = ind.hidden, ind.lag
	newInd.randGen = ind.randGen
	return &newInd
}

// GetWeights ...
func (ind *Individual) GetWeights() [][]float64 {
	return ind.weights
}

// GetMutationSteps ...
func (ind *Individual) GetMutationSteps() [][]float64 {
	return ind.mSteps
}

// GetLag ...
func (ind *Individual) GetLag() int {
	return ind.lag
}

// Fitness ...
func (ind *Individual) Fitness(input [][]float64, target []float64) float64 {
	output := ind.Predict(input)
	return ind.fitnessFunction(target, output)
}

// Predict ...
func (ind *Individual) Predict(input [][]float64) []float64 {
	prediction := make([]float64, len(input))
	for index, value := range input {
		prediction[index] = ind.predictSolo(value)
	}
	return prediction
}

// Mutate ...
func (ind *Individual) Mutate() *Individual{
	newInd := ind.copy()
	for i := 0; i <= newInd.lag; i++ {
		for j := 0; j < newInd.hidden; j++ {
			newInd.mutateMS(i, j)
			newInd.mutateWeight(i, j)
		}
	}

	for i := 0; i <= newInd.hidden; i++ {
		newInd.mutateMS(i, -1)
		newInd.mutateWeight(i, -1)
	}

	return newInd
}

// NewIndividual ...
func NewIndividual(lag, hidden int, randGen * rand.Rand, fitFunc func(t, o []float64) float64) (*Individual) {
	weights := make([][]float64, lag + 1)
	mSteps := make([][]float64, lag + 1)
	biasWeights := make([]float64, hidden + 1)
	biasMSteps := make([]float64, hidden + 1)
	wTotal := float64(lag * hidden)
	t := 1.0/(math.Sqrt(2.0 * math.Sqrt(wTotal)))
	tLine := 1.0/(math.Sqrt(2.0 * wTotal))

	for i := 0; i <= lag; i++ {
		weights[i] = make([]float64, hidden)
		mSteps[i] = make([]float64, hidden)
		for j := 0; j < hidden; j++ {
			weights[i][j] = math.Max(randGen.NormFloat64() * 0.5/3.0 + 0.5, 0.0)
			weights[i][j] = math.Min(weights[i][j], 1.0)
			mSteps[i][j] = randGen.Float64()
		}
	}
	for j := 0; j <= hidden; j++ {
		biasWeights[j] = math.Max(randGen.NormFloat64() * 0.5/3.0 + 0.5, 0.0)
		biasWeights[j] = math.Min(biasWeights[j], 1.0)
		biasMSteps[j] = randGen.Float64()
	}

	return &Individual{
		weights: weights,
		mSteps: mSteps,
		biasWeights: biasWeights,
		biasMSteps: biasMSteps,
		t:t,
		tLine: tLine,
		hidden: hidden,
		lag: lag,
		randGen: randGen,
		fitnessFunction: fitFunc,
	}
}